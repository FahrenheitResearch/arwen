"""Saved setups: the library copy, the re-timed start, and every refusal."""
import contextlib
import copy
from datetime import datetime, timedelta
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tomllib

import pytest

from gpuwm import cli, companion_domains as editor, companion_setups as setups
from gpuwm import fetch, fetch_routes
from gpuwm.hrrr_prepared_bundle import render_wps_namelist
from gpuwm.namelist_import import parse_namelist_text
from gpuwm.static import rust_bridge
from gpuwm.toml_document import emit_experiment_toml

from test_companion_domains import configured_case

#: Both doors in this deck republish the configuration through
#: gpuwm.companion_domains, which refuses with "Domain editing requires the
#: native static-fields bridge" when that library is not built.  The gate is
#: module level because save and start share that one route, and it is
#: declared in tools/battery/must_run_gates.txt so the coverage that leaves
#: on a bridgeless box is stated rather than silent.
_STATIC_BRIDGE_UNAVAILABLE = rust_bridge.unavailable_reason()

pytestmark = pytest.mark.skipif(
    _STATIC_BRIDGE_UNAVAILABLE is not None,
    reason=("the native static-fields bridge is not built, and every saved "
            "setup is published through it: "
            f"{_STATIC_BRIDGE_UNAVAILABLE}"))


SAVED_NAME = "Front range 4 km"
SAVED_SLUG = "front-range-4-km"


def gfs_case(tmp_path):
    """The domain fixture as a downloadable GFS forecast with its WPS companion."""
    source, raw = configured_case(tmp_path)
    raw = copy.deepcopy(raw)
    raw.pop("case_data")
    raw["fetch"] = {"source": "gfs", "cycle": "2013-05-31T18", "hours": 6,
                    "cadence": 3, "out": "data/proof"}
    source.write_text(emit_experiment_toml(raw), encoding="utf-8")
    (tmp_path / "namelist.wps").unlink()
    exp = editor._build(raw, source)
    source.with_suffix(".namelist.wps").write_text(
        render_wps_namelist(exp), encoding="utf-8")
    return source, raw


def era5_case(tmp_path):
    """The same fixture as an ERA5 case with its declared Vtable and WPS."""
    source, raw = configured_case(tmp_path)
    raw = copy.deepcopy(raw)
    raw["fetch"] = {"source": "era5", "cycle": "2013-05-31T18", "hours": 6,
                    "cadence": 6, "area": "34.5,-98.5,35.5,-97.5"}
    source.write_text(emit_experiment_toml(raw), encoding="utf-8")
    (tmp_path / "Vtable.ERA5").write_text("GRIB | Level |\n", encoding="utf-8")
    return source, raw


def route_case(tmp_path):
    """A real configuration authored by the domain door, route companions and all."""
    out = tmp_path / "proof.toml"
    args = cli.build_parser().parse_args([
        "domain", "--source", "hrrr", "--point", "39.0,-98.0",
        "--cycle", "2026-09-09T12", "--hours", "3", "--root-dx", "12",
        "--vram-gib", "12", "--history-interval", "900", "--out", str(out)])
    with contextlib.redirect_stdout(io.StringIO()):
        assert args.func(args) == 0
    return out


def save(source, library, name=SAVED_NAME):
    return setups.save_setup(config_path=source, library=library, name=name)


def saved_tables(folder):
    return tomllib.loads((folder / "setup.toml").read_text(encoding="utf-8"))


def metadata(folder):
    return json.loads((folder / "setup.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------

def test_save_copies_the_configuration_companion_and_metadata(tmp_path):
    source, original = gfs_case(tmp_path)
    before = source.read_bytes()
    library = tmp_path / "setups"
    result = save(source, library)
    folder = library / SAVED_SLUG
    assert sorted(path.name for path in folder.iterdir()) == [
        "setup.json", "setup.namelist.wps", "setup.toml"]
    assert source.read_bytes() == before
    saved = saved_tables(folder)
    for table in ("shared", "domain", "projection", "experiment"):
        assert saved[table] == original[table]
    assert saved.get("tiles") == original.get("tiles")
    assert saved["experiment"]["restart_interval_s"] == original["experiment"]["restart_interval_s"]
    assert saved["fetch"]["out"] == str(Path("data/proof").expanduser().resolve())
    assert {change["field"] for change in result["changes"]} == {"fetch.out"}
    document = metadata(folder)
    assert document["schema"] == "arwen.saved-setup.v1"
    assert (document["name"], document["slug"]) == (SAVED_NAME, SAVED_SLUG)
    assert document["setup_path"] == str(folder / "setup.toml")
    assert document["setup_sha256"] == hashlib.sha256(
        (folder / "setup.toml").read_bytes()).hexdigest()
    assert document["source_config_path"] == str(source)
    assert document["source_config_sha256"] == hashlib.sha256(before).hexdigest()
    assert document["fetch_source"] == "gfs"
    assert document["forecast_start_hour"] == 0
    assert document["run_seconds"] == original["experiment"]["run_seconds"]
    assert document["files"] == ["setup.json", "setup.namelist.wps", "setup.toml"]
    assert [row["grid_id"] for row in document["domains"]] == [1, 2, 3]
    assert document["domains"][0]["dx_m"] == 12000.0
    assert (result["schema"], result["action"]) == (setups.RESULT_SCHEMA, "save")
    assert result["created"] is True and result["forecast_started"] is False
    assert result["setup"] == document
    assert parse_namelist_text((folder / "setup.namelist.wps").read_text())


def test_summary_derives_from_native_geometry_and_the_registry(tmp_path):
    source, original = gfs_case(tmp_path)
    library = tmp_path / "setups"
    summary = save(source, library)["setup"]["summary"]
    assert "d01 12 km 180 x 160" in summary
    assert "d02 4 km 72 x 72" in summary
    microphysics = next(component for component in editor.physics_components()
                        if component["id"] == "microphysics")
    declared = original["shared"].get("mp_physics", 0)
    option = next((row for row in microphysics["options"]
                   if row["selectors"] == {"mp_physics": declared}), None)
    if option is not None:
        assert option["label"] in summary
    from gpuwm.source_adapters import get_source_adapter
    assert get_source_adapter("gfs").display_title in summary
    assert "output every 60 s" in summary
    assert "tiles off" in summary


def test_summary_names_selectors_verbatim_when_no_option_matches():
    """No invented scheme name: an unmatched component prints its selectors."""
    component = next(row for row in editor.physics_components()
                     if any(option["selectors"] for option in row["options"]))
    key = sorted({key for option in component["options"]
                  for key in option["selectors"]})[0]
    taken = {option["selectors"].get(key) for option in component["options"]}
    value = next(candidate for candidate in range(2000) if candidate not in taken)
    exp = SimpleNamespace(
        root=SimpleNamespace(run=SimpleNamespace(**{key: value}),
                             history_interval_s=900.0),
        tiles=SimpleNamespace(mode="auto"))
    line = setups.summary_line(exp, {}, [])
    assert f"{key}={value}" in line
    assert line.endswith("output every 900 s · tiles auto")


@pytest.mark.parametrize("name, sentence", [
    ("", setups.NAME_REFUSAL),
    ("   ", setups.NAME_REFUSAL),
    ("a/b", setups.NAME_REFUSAL),
    ("a\\b", setups.NAME_REFUSAL),
    ('a"b', setups.NAME_REFUSAL),
    ("a:b", setups.NAME_REFUSAL),
    ("x" * 81, setups.NAME_REFUSAL),
    ("..", setups.NAME_REFUSAL),
])
def test_save_name_rules_publish_nothing(tmp_path, name, sentence):
    source, _original = gfs_case(tmp_path)
    library = tmp_path / "setups"
    with pytest.raises(ValueError) as caught:
        save(source, library, name)
    assert str(caught.value) == sentence
    assert not library.exists()


def test_save_collision_keeps_the_first_setup(tmp_path):
    source, _original = gfs_case(tmp_path)
    library = tmp_path / "setups"
    save(source, library)
    folder = library / SAVED_SLUG
    before = (folder / "setup.toml").read_bytes()
    with pytest.raises(ValueError) as caught:
        save(source, library, "FRONT   RANGE 4 KM")
    assert str(caught.value) == (
        f'A saved setup named "{SAVED_NAME}" already exists in this library. '
        "Choose another name, or delete that setup first.")
    assert sorted(path.name for path in library.iterdir()) == [SAVED_SLUG]
    assert (folder / "setup.toml").read_bytes() == before


def test_save_refuses_a_forecast_without_its_wps_companion(tmp_path):
    source, _original = gfs_case(tmp_path)
    source.with_suffix(".namelist.wps").unlink()
    with pytest.raises(ValueError) as caught:
        save(source, tmp_path / "setups")
    assert str(caught.value) == (
        "This forecast has no source.namelist.wps beside it, so the staged "
        "route could not run a forecast started from it. Create the forecast "
        "again, then save it.")
    assert not (tmp_path / "setups").exists()


def test_save_refuses_an_unreadable_configuration(tmp_path):
    with pytest.raises(ValueError) as caught:
        setups.save_setup(config_path=tmp_path / "gone.toml",
                          library=tmp_path / "setups", name=SAVED_NAME)
    assert str(caught.value).endswith(" Open the forecast again, then save it.")
    assert not (tmp_path / "setups").exists()


def test_save_copies_case_data_files_and_rewrites_them_relative(tmp_path):
    source, original = era5_case(tmp_path)
    library = tmp_path / "setups"
    save(source, library)
    folder = library / SAVED_SLUG
    assert (folder / "Vtable.ERA5").read_text() == (tmp_path / "Vtable.ERA5").read_text()
    saved = saved_tables(folder)
    assert saved["case_data"]["vtable"] == "Vtable.ERA5"
    assert saved["case_data"]["wps_namelist"] == "setup.namelist.wps"
    assert saved["case_data"]["forcing"] == original["case_data"]["forcing"]
    assert saved["case_data"]["geog_root"] == original["case_data"]["geog_root"]
    assert (folder / "setup.namelist.wps").read_text() == (
        tmp_path / "namelist.wps").read_text()
    assert set(metadata(folder)["files"]) == {
        "Vtable.ERA5", "setup.json", "setup.namelist.wps", "setup.toml"}


# ---------------------------------------------------------------------------
# start
# ---------------------------------------------------------------------------

def start(folder, tmp_path, **overrides):
    request = dict(setup_path=folder / "setup.toml", cycle="2013-05-31T12",
                   hours=7.0, forecast_start_hour=3, name="Restarted forecast",
                   out=tmp_path / "forecasts" / "restart.toml")
    request.update(overrides)
    return setups.start_setup(**request)


def test_start_changes_only_the_timed_fields(tmp_path):
    source, original = gfs_case(tmp_path)
    library = tmp_path / "setups"
    save(source, library)
    folder = library / SAVED_SLUG
    saved = saved_tables(folder)
    result = start(folder, tmp_path)
    out = Path(result["config_path"])
    written = tomllib.loads(out.read_text(encoding="utf-8"))
    assert written["experiment"]["name"] == "Restarted forecast"
    assert written["experiment"]["start_time"] == datetime(2013, 5, 31, 15)
    assert written["experiment"]["run_seconds"] == 25200.0
    assert written["fetch"]["cycle"] == "2013-05-31T12"
    assert written["fetch"]["hours"] == 9
    assert written["fetch"]["forecast_start_hour"] == 3
    assert written["fetch"]["out"] == str(out.parent / "data" / "Restarted forecast")
    for table in ("shared", "domain", "projection"):
        assert written[table] == saved[table]
    assert written.get("tiles") == saved.get("tiles")
    assert written["experiment"]["restart_interval_s"] == saved["experiment"]["restart_interval_s"]
    assert {change["field"] for change in result["changes"]} == {
        "experiment.name", "experiment.start_time", "experiment.run_seconds",
        "fetch.cycle", "fetch.hours", "fetch.forecast_start_hour", "fetch.out"}
    wps = parse_namelist_text(Path(result["wps_path"]).read_text())
    assert wps["share"]["start_date"][0] == "2013-05-31_15:00:00"
    assert wps["share"]["end_date"][0] == "2013-05-31_22:00:00"
    assert result["timing"] == {
        "source": "gfs", "cycle": "2013-05-31T12", "forecast_start_hour": 3,
        "hours": 7.0, "fetch_hours": 9, "start_time": "2013-05-31T15:00:00",
        "cadence_hours": 3}
    assert result["setup_sha256"] == metadata(folder)["setup_sha256"]
    assert result["setup_name"] == SAVED_NAME
    assert Path(result["receipt_path"]).is_file()
    assert result["created"] is True
    assert result["forecast_started"] is False and result["acquisition_started"] is False
    from gpuwm.companion_query import inspect_configuration
    assert (json.loads(editor._json(result["configuration"])) ==
            json.loads(editor._json(inspect_configuration(out))))
    # The saved setup itself is untouched by starting from it.
    assert saved_tables(folder) == saved


def test_start_latest_resolves_through_the_source_grid(tmp_path, monkeypatch):
    source, _original = gfs_case(tmp_path)
    library = tmp_path / "setups"
    save(source, library)
    seen = {}

    def resolve(source_id, last_hour, **kwargs):
        seen.update(source=source_id, last_hour=last_hour)
        return datetime(2013, 5, 31, 6)

    monkeypatch.setattr(fetch, "resolve_latest_cycle", resolve)
    result = start(library / SAVED_SLUG, tmp_path, cycle="latest")
    assert seen == {"source": "gfs", "last_hour": 12}
    assert result["timing"]["cycle"] == "2013-05-31T06"
    assert tomllib.loads(Path(result["config_path"]).read_text())[
        "experiment"]["start_time"] == datetime(2013, 5, 31, 9)


def _table_route_source():
    """A registered table route whose producer does not run every hour."""
    for source_id in fetch_routes.route_ids():
        if source_id == "era5":
            continue
        route = fetch_routes.route_for(source_id)
        if 0 < len(route.cycle_hours) < 24:
            missing = next(hour for hour in range(24)
                           if hour not in route.cycle_hours)
            return source_id, route, missing
    pytest.skip("no table route declares a restricted cycle-hour grid")


def test_start_refuses_a_cycle_hour_a_table_route_does_not_run(tmp_path):
    source, raw = gfs_case(tmp_path)
    source_id, route, missing = _table_route_source()
    raw["fetch"] = {"source": source_id, "cycle": f"2013-05-31T{route.cycle_hours[0]:02d}",
                    "hours": route.default_cadence, "cadence": route.default_cadence,
                    "out": "data/proof"}
    source.write_text(emit_experiment_toml(raw), encoding="utf-8")
    library = tmp_path / "setups"
    save(source, library)
    with pytest.raises(ValueError) as caught:
        start(library / SAVED_SLUG, tmp_path, cycle=f"2013-05-31T{missing:02d}",
              hours=float(route.default_cadence), forecast_start_hour=0)
    assert "is not a cycle this producer runs" in str(caught.value)
    assert not (tmp_path / "forecasts" / "restart.toml").exists()


@pytest.mark.parametrize("overrides, expected", [
    ({"name": "  "}, setups.FORECAST_NAME_REFUSAL),
    ({"hours": 0}, setups.HOURS_REFUSAL),
    ({"hours": -3.0}, setups.HOURS_REFUSAL),
    ({"forecast_start_hour": -1}, setups.LEAD_REFUSAL),
    ({"cycle": "2013-05-31T05"}, "GFS cycles run at 00/06/12/18 UTC only"),
    ({"cycle": "yesterday"}, "must be YYYY-MM-DDTHH (UTC) or 'latest'"),
    ({"hours": 400.0, "forecast_start_hour": 0}, "384"),
])
def test_start_refusals_publish_nothing(tmp_path, overrides, expected):
    source, _original = gfs_case(tmp_path)
    library = tmp_path / "setups"
    save(source, library)
    with pytest.raises(ValueError) as caught:
        start(library / SAVED_SLUG, tmp_path, **overrides)
    assert expected in str(caught.value)
    out = tmp_path / "forecasts" / "restart.toml"
    for path in (out, out.with_suffix(".namelist.wps"),
                 out.with_suffix(".setup-start.json")):
        assert not path.exists()


def test_start_refuses_an_existing_output(tmp_path):
    source, _original = gfs_case(tmp_path)
    library = tmp_path / "setups"
    save(source, library)
    out = tmp_path / "forecasts" / "restart.toml"
    out.parent.mkdir(parents=True)
    out.write_text("# already here\n", encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        start(library / SAVED_SLUG, tmp_path)
    assert str(caught.value) == (
        f"A configuration already exists at {out}. "
        "Choose a new name, or use Open configuration.")
    assert out.read_text(encoding="utf-8") == "# already here\n"


def test_start_refuses_a_tampered_setup(tmp_path):
    source, _original = gfs_case(tmp_path)
    library = tmp_path / "setups"
    save(source, library)
    folder = library / SAVED_SLUG
    (folder / "setup.toml").write_text(
        (folder / "setup.toml").read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        start(folder, tmp_path)
    assert str(caught.value) == (
        f'The saved setup "{SAVED_NAME}" was changed after it was saved: '
        "setup.toml no longer matches setup.json. Delete it from Saved "
        "setups and save the forecast again.")
    assert not (tmp_path / "forecasts").exists()


def test_start_refuses_an_unreadable_setup(tmp_path):
    with pytest.raises(ValueError) as caught:
        start(tmp_path / "gone", tmp_path)
    assert str(caught.value).startswith("The saved setup cannot be read: ")
    assert str(caught.value).endswith(
        " Delete it from Saved setups and save the forecast again.")


def test_start_refuses_a_setup_with_no_forcing_source(tmp_path):
    source, raw = gfs_case(tmp_path)
    raw.pop("fetch")
    source.write_text(emit_experiment_toml(raw), encoding="utf-8")
    library = tmp_path / "setups"
    save(source, library)
    with pytest.raises(ValueError) as caught:
        start(library / SAVED_SLUG, tmp_path)
    assert str(caught.value) == (
        "This saved setup declares no forcing source, so there is no cycle to "
        "start it from. Save it from a forecast created in Create forecast.")
    assert not (tmp_path / "forecasts").exists()


def test_route_companions_are_saved_and_rendered_again_at_the_new_cycle(tmp_path):
    """A copied route input would carry the saved forecast's dates into the run."""
    source = route_case(tmp_path)
    library = tmp_path / "setups"
    document = save(source, library)["setup"]
    folder = library / SAVED_SLUG
    assert set(document["files"]) == {
        "setup.toml", "setup.json", "setup.namelist.wps",
        "setup.namelist.input", "setup.stock.namelist.input",
        "setup.d01-target.json"}
    saved_input = (folder / "setup.namelist.input").read_text(encoding="utf-8")
    result = start(folder, tmp_path, cycle="2026-09-08T06", hours=3.0,
                   forecast_start_hour=0)
    out = Path(result["config_path"])
    for name in ("namelist.wps", "namelist.input", "stock.namelist.input"):
        assert out.with_suffix("." + name).is_file()
    assert (out.parent / "restart.d01-target.json").is_file()
    started_input = out.with_suffix(".namelist.input").read_text(encoding="utf-8")
    assert started_input != saved_input
    assert " start_day                          = 08," in started_input
    assert " start_hour                         = 06," in started_input
    assert " start_day                          = 09," in saved_input
    wps = parse_namelist_text(out.with_suffix(".namelist.wps").read_text())
    assert wps["share"]["start_date"][0] == "2026-09-08_06:00:00"


def test_start_era5_delegates_to_the_forcing_editor(tmp_path):
    source, _original = era5_case(tmp_path)
    library = tmp_path / "setups"
    save(source, library)
    result = start(library / SAVED_SLUG, tmp_path, cycle="2013-05-31T12",
                   forecast_start_hour=0, hours=6.0)
    written = tomllib.loads(Path(result["config_path"]).read_text(encoding="utf-8"))
    assert written["fetch"]["source"] == "era5"
    assert written["fetch"]["retrieve"] is True
    assert written["fetch"]["cadence"] == 6
    assert ".forcing-cache" in written["case_data"]["forcing"][0]
    assert written["experiment"]["start_time"] == datetime(2013, 5, 31, 12)
    assert result["configuration"]["schema"] == "arwen.companion-configuration.v1"
    assert result["configuration"]["geometry_backend"] == "rust-static-fields"
    assert Path(result["receipt_path"]).is_file()
    assert result["receipt_path"] == str(
        Path(result["config_path"]).with_suffix(".forcing.json"))
    assert result["acquisition_started"] is False


def test_start_era5_carries_the_vtable_beside_the_new_forecast_and_keeps_the_saved_wps(tmp_path):
    """A started forecast must outlive its setup's deletion and keep the WPS choices."""
    source, _original = era5_case(tmp_path)
    library = tmp_path / "setups"
    save(source, library)
    folder = library / SAVED_SLUG
    saved_wps = parse_namelist_text((folder / "setup.namelist.wps").read_text())
    result = start(folder, tmp_path, cycle="2013-05-31T12",
                   forecast_start_hour=0, hours=6.0)
    out = Path(result["config_path"])
    written = tomllib.loads(out.read_text(encoding="utf-8"))
    carried = out.parent / "Vtable.ERA5"
    assert carried.is_file()
    assert carried.read_text() == (folder / "Vtable.ERA5").read_text()
    assert written["case_data"]["vtable"] == str(carried)
    assert written["case_data"]["wps_namelist"] == str(out.with_suffix(".namelist.wps"))
    started_wps = parse_namelist_text(out.with_suffix(".namelist.wps").read_text())
    assert started_wps["geogrid"]["geog_data_res"] == saved_wps["geogrid"]["geog_data_res"]
    assert started_wps["share"]["start_date"][0] == "2013-05-31_12:00:00"
    # Deleting the setup afterwards leaves every declared input of the new forecast in place.
    import shutil
    shutil.rmtree(folder)
    assert Path(written["case_data"]["vtable"]).is_file()
    assert Path(written["case_data"]["wps_namelist"]).is_file()
    # A second start into the same folder reuses the identical Vtable already there.
    save(source, library)
    again = start(folder, tmp_path, cycle="2013-05-31T12", forecast_start_hour=0,
                  hours=6.0, name="Second restart",
                  out=tmp_path / "forecasts" / "second.toml")
    second = tomllib.loads(Path(again["config_path"]).read_text(encoding="utf-8"))
    assert second["case_data"]["vtable"] == str(carried)
    assert sorted(path.name for path in out.parent.glob("Vtable*")) == ["Vtable.ERA5"]


# ---------------------------------------------------------------------------
# The door itself
# ---------------------------------------------------------------------------

def test_cli_registration_round_trips_every_flag(tmp_path):
    parser = cli.build_parser()
    saved = parser.parse_args(["companion-setups", "save", "--config", "a.toml",
                               "--library", "lib", "--name", "Front range"])
    assert (saved.setups_action, saved.name) == ("save", "Front range")
    assert (saved.config, saved.library) == (Path("a.toml"), Path("lib"))
    assert saved.func is setups.main
    started = parser.parse_args([
        "companion-setups", "start", "setup.toml", "--cycle", "latest",
        "--hours", "9", "--forecast-start-hour", "3", "--name", "New",
        "--out", "new.toml"])
    assert (started.setups_action, started.cycle, started.hours) == ("start", "latest", 9.0)
    assert (started.forecast_start_hour, started.name) == (3, "New")
    assert (started.setup, started.out) == (Path("setup.toml"), Path("new.toml"))
    assert started.func is setups.main


def test_the_options_page_documents_both_actions():
    page = (Path(__file__).resolve().parents[1] / "docs" / "public" /
            "CLI-OPTIONS.md").read_text(encoding="utf-8")
    assert "`gpuwm companion-setups save`" in page
    assert "`gpuwm companion-setups start`" in page


def test_main_prints_one_refusal_document_and_exits_one(tmp_path, capsys):
    args = cli.build_parser().parse_args([
        "companion-setups", "save", "--config", str(tmp_path / "gone.toml"),
        "--library", str(tmp_path / "setups"), "--name", "x"])
    assert setups.main(args) == 1
    document = json.loads(capsys.readouterr().out)
    assert document["schema"] == setups.RESULT_SCHEMA
    assert document["created"] is False and document["forecast_started"] is False
    assert document["acquisition_started"] is False
    assert document["error"].endswith(" Open the forecast again, then save it.")
    assert not (tmp_path / "setups").exists()


def test_round_trip_save_start_validate(tmp_path):
    from gpuwm.experiment import experiment_config_document, load_experiment
    source, _original = gfs_case(tmp_path)
    library = tmp_path / "setups"
    save(source, library)
    folder = library / SAVED_SLUG
    result = start(folder, tmp_path)
    saved_exp = load_experiment(folder / "setup.toml")
    started_exp = load_experiment(Path(result["config_path"]))
    def without_the_clock(document):
        document = copy.deepcopy(document)
        for key in ("name", "start_time", "run_seconds", "end_time", "source"):
            document.pop(key, None)
        for row in document.get("domains", []):
            row.pop("start_time", None)
            row.get("run", {}).pop("run_seconds", None)
        return document

    before = experiment_config_document(saved_exp)
    after = experiment_config_document(started_exp)
    assert without_the_clock(before) == without_the_clock(after)
    assert before["name"] != after["name"]
    assert (editor.native_domain_outlines(saved_exp) ==
            editor.native_domain_outlines(started_exp))
    assert started_exp.start_time == datetime(2013, 5, 31, 15)
    assert started_exp.run_seconds == 25200.0
    assert started_exp.start_time - timedelta(hours=3) == datetime(2013, 5, 31, 12)
