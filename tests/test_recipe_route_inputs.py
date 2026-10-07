"""A shipped recipe runs from its own TOML: the WPS namelist and the static file.

Two breakages, both found by the WOOF 1.0.3 GPU smoke on box B (2026-10-06)
and both present in 2.8.6:

* ``gpuwm go`` on any of the four HRRR configuration recipes in
  ``configs/recipes`` stopped before any stage, "the staged route reads
  hrrr_configuration_cut.namelist.wps beside ...": none of them carried a
  WPS twin.  The route now renders the namelist from the configuration's
  own grid when none sits beside it (``configuration_wps_namelist``).
* A recipe naming ``[static] source = "hrrr-conus-v4"`` stopped at static
  preparation, after the forcing download, until ``fetch-geog
  --static-source hrrr-conus-v4`` had been run by hand.  The run now
  stages the file through that command's own fetch, with a size line and
  progress lines (``stage_required_static_sources``).
"""
from __future__ import annotations

import io
import shutil
import tomllib
from pathlib import Path

import pytest

from gpuwm import go_cli, runplan
from gpuwm.companion_domains import configuration_wps_namelist
from gpuwm.experiment import load_experiment
from gpuwm.geog_assets import download_archive
from gpuwm.namelist_import import parse_namelist_text
from gpuwm.source_adapters import source_forcing_interval_seconds
from gpuwm.native_wrf_contract import validate_wps_geometry
from gpuwm.static import external_source
from gpuwm.static.external_source import (required_static_rows,
                                          stage_required_static_sources,
                                          static_source_row)

ROOT = Path(__file__).resolve().parents[1]
RECIPES = sorted((ROOT / "configs" / "recipes").glob("*.toml"))


def _raw(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8-sig"))


def _shipped_configs_without_a_twin() -> list[Path]:
    """Every shipped config a door runs by path and no WPS twin sits beside."""
    found = []
    for path in sorted((ROOT / "configs").rglob("*.toml")):
        try:
            raw = _raw(path)
        except (tomllib.TOMLDecodeError, UnicodeDecodeError):
            continue
        if "fetch" not in raw or "domain" not in raw or "case_data" in raw:
            continue
        if path.with_name(f"{path.stem}.namelist.wps").is_file():
            continue
        found.append(path)
    return found


def test_the_recipes_are_the_ones_this_file_was_written_for():
    # A recipe added later is covered by the parametrized cells below; this
    # cell only says the directory is not empty, so they cannot all skip.
    assert {path.stem for path in RECIPES} >= {
        "conus_hrrr_configuration", "hrrr_configuration_clock",
        "hrrr_configuration_cut", "hrrr_v4_gsd41"}


@pytest.mark.parametrize("recipe", RECIPES, ids=lambda p: p.stem)
def test_every_shipped_recipe_renders_a_namelist_that_is_its_grid(recipe, tmp_path):
    raw = _raw(recipe)
    exp = load_experiment(recipe)
    written = configuration_wps_namelist(recipe, exp, raw=raw, into=tmp_path)
    assert written == tmp_path / f"{recipe.stem}.namelist.wps"
    # The forecast stage's own geometry comparison, on the bytes the run binds.
    validate_wps_geometry(exp, written, source_name="recipe")
    tables = parse_namelist_text(written.read_text(encoding="utf-8"))
    geogrid, share = tables["geogrid"], tables["share"]
    domain = raw["domain"][0]
    assert geogrid["e_we"] == [domain["nx"] + 1]
    assert geogrid["e_sn"] == [domain["ny"] + 1]
    assert geogrid["dx"] == [domain["dx"]]
    for key in ("ref_lat", "ref_lon", "truelat1", "truelat2", "stand_lon"):
        assert geogrid[key] == [raw["projection"][key]], key
    assert geogrid["map_proj"] == [raw["projection"]["map_proj"]]
    # The boundary interval is the cadence the recipe fetches at.  The
    # hand-kept twins the WOOF 1.0.3 assembly shipped said 10800 for the
    # cut after the recipe had moved to hourly boundaries.
    cadence = raw["fetch"].get("cadence")
    expected = (cadence * 3600 if cadence is not None
                else int(source_forcing_interval_seconds(raw["fetch"]["source"])))
    assert share["interval_seconds"] == [expected]


@pytest.mark.parametrize("config", _shipped_configs_without_a_twin(),
                         ids=lambda p: p.relative_to(ROOT).as_posix())
def test_no_shipped_config_without_a_twin_is_left_unrenderable(config, tmp_path):
    """The gap search, kept: every shipped config with no WPS twin renders one.

    The search for the recipes' gap found more configs with no twin (the
    HRRR-source LES and demo configs, an hrrr-prs vertical-order config, a
    starter template and a tiles control leg).  Each one either renders a
    namelist matching its grid, or does not load as an experiment at all.
    """
    try:
        exp = load_experiment(config)
    except Exception as error:  # noqa: BLE001 - a config that does not load has no grid
        pytest.skip(f"does not load as an experiment: {type(error).__name__}")
    written = configuration_wps_namelist(config, exp, raw=_raw(config), into=tmp_path)
    validate_wps_geometry(exp, written, source_name="shipped")


def test_a_namelist_beside_the_config_is_read_and_left_alone(tmp_path):
    config = tmp_path / "case.toml"
    shutil.copy(ROOT / "configs" / "recipes" / "hrrr_configuration_cut.toml", config)
    beside = tmp_path / "case.namelist.wps"
    beside.write_text("&share\n wrf_core = 'ARW',\n/\n", encoding="utf-8")
    before = beside.read_bytes()
    found = configuration_wps_namelist(config, load_experiment(config),
                                       raw=_raw(config), into=tmp_path / "run")
    assert found == beside
    assert beside.read_bytes() == before
    assert not (tmp_path / "run").exists()


def test_the_door_question_renders_and_writes_nothing(tmp_path):
    config = tmp_path / "case.toml"
    shutil.copy(ROOT / "configs" / "recipes" / "hrrr_configuration_cut.toml", config)
    assert configuration_wps_namelist(config, load_experiment(config),
                                      raw=_raw(config)) is None
    assert sorted(p.name for p in tmp_path.iterdir()) == ["case.toml"]


def test_a_config_whose_grid_cannot_be_written_is_refused_by_name(tmp_path, monkeypatch):
    config = tmp_path / "case.toml"
    shutil.copy(ROOT / "configs" / "recipes" / "hrrr_configuration_cut.toml", config)
    import gpuwm.companion_domains as companion

    monkeypatch.setattr(companion, "candidate_wps_text",
                        lambda *a, **k: "&share\n max_dom = 1,\n/\n&geogrid\n"
                        " e_we = 9,\n e_sn = 9,\n dx = 3000.0,\n dy = 3000.0,\n"
                        " map_proj = 'lambert',\n ref_lat = 38.5,\n ref_lon = -97.5,\n"
                        " truelat1 = 38.5,\n truelat2 = 38.5,\n stand_lon = -97.5,\n/\n")
    with pytest.raises(ValueError, match="does not match its own grid"):
        configuration_wps_namelist(config, load_experiment(config),
                                   raw=_raw(config), into=tmp_path / "run")
    assert not list((tmp_path / "run").glob("*.namelist.wps"))
    assert not list((tmp_path / "run").glob("*.tmp"))


def test_the_staged_door_passes_a_shipped_recipe_with_no_twin(tmp_path):
    """`gpuwm go --dry-run` / plan review on the cut: no namelist refusal."""
    config = tmp_path / "hrrr_configuration_cut.toml"
    shutil.copy(ROOT / "configs" / "recipes" / "hrrr_configuration_cut.toml", config)
    raw = {"schema": runplan.PLAN_SCHEMA, "name": "cut", "route": "prepared",
           "config": {"path": str(config)}, "output_root": str(tmp_path / "out"),
           "run_options": {"geog_root": str(tmp_path / "geog")}}
    plan = runplan.build_plan(raw, source="test", base_dir=tmp_path, sha256="0" * 64)
    resolution = runplan.resolve_plan(plan, require_inputs=False)
    assert resolution is not None
    assert not list(tmp_path.glob("*.namelist.wps"))


def test_the_staged_chain_binds_the_rendered_namelist(tmp_path, monkeypatch):
    """The staged chain hands the preparation the namelist it rendered."""
    config = tmp_path / "hrrr_v4_gsd41.toml"
    shutil.copy(ROOT / "configs" / "recipes" / "hrrr_v4_gsd41.toml", config)
    exp = load_experiment(config)
    run = tmp_path / "run"
    seen = {}

    class Stop(Exception):
        pass

    def fetch(*a, **k):
        raise Stop

    monkeypatch.setattr(runplan, "_run_fetch", fetch)
    monkeypatch.setattr(runplan, "_native_fetch_beside", lambda *a, **k: None,
                        raising=False)

    def record(raw, geog_root, *, base_dir):
        seen["static"] = (raw["fetch"]["source"], Path(geog_root))

    monkeypatch.setattr(runplan, "_stage_static_sources", record)
    from types import SimpleNamespace

    plan = SimpleNamespace(run_options={"data_dir": str(tmp_path / "data"),
                                        "geog_root": str(tmp_path / "geog")},
                           config_intent={})
    observer = SimpleNamespace(enter_stage=lambda *a, **k: None,
                               finish_stage=lambda **k: None,
                               warn=lambda *a, **k: None,
                               arm_first_products=lambda *a, **k: None)
    with pytest.raises(Stop):
        runplan._staged_chain(plan, config_path=config, exp=exp,
                              observer=observer, run_dir=run)
    rendered = run / "chain" / "route-inputs" / "hrrr_v4_gsd41.namelist.wps"
    assert rendered.is_file()
    validate_wps_geometry(exp, rendered, source_name="recipe")
    assert seen["static"] == ("rap-native", tmp_path / "geog")
    assert not (tmp_path / "hrrr_v4_gsd41.namelist.wps").exists()


def test_the_go_chain_renders_into_the_claimed_run_folder(tmp_path):
    config = tmp_path / "gfs_case.toml"
    shutil.copy(ROOT / "configs" / "gfs_12km_quickstart.toml", config)
    plan = go_cli.plan_from_config(config, outdir=tmp_path / "out")
    assert plan["wps_namelist_rendered"] is True
    assert not Path(plan["wps_namelist"]).exists()
    go_cli.claim_run_root(plan)
    written = Path(plan["wps_namelist"])
    assert written == Path(plan["root"]) / "route-inputs" / "gfs_case.namelist.wps"
    validate_wps_geometry(load_experiment(config), written, source_name="gfs")
    assert "--base-wps-namelist" in go_cli.authority_command(plan)
    assert str(written) in go_cli.authority_command(plan)


def test_the_go_chain_keeps_a_twin_beside_the_config(tmp_path):
    config = tmp_path / "gfs_case.toml"
    shutil.copy(ROOT / "configs" / "gfs_12km_quickstart.toml", config)
    shutil.copy(ROOT / "configs" / "gfs_12km_quickstart.namelist.wps",
                tmp_path / "gfs_case.namelist.wps")
    plan = go_cli.plan_from_config(config, outdir=tmp_path / "out")
    assert plan["wps_namelist_rendered"] is False
    go_cli.claim_run_root(plan)
    assert plan["wps_namelist"] == tmp_path / "gfs_case.namelist.wps"


# ---------------------------------------------------------------------------
# The static file.
# ---------------------------------------------------------------------------

def test_the_recipes_that_read_the_hrrr_static_file_are_the_ones_that_name_it():
    wanted = {path.stem: [row.id for row in required_static_rows(
        _raw(path), base_dir=path.parent)] for path in RECIPES}
    assert wanted == {
        "conus_hrrr_configuration": ["hrrr-conus-v4"],
        "hrrr_configuration_clock": ["hrrr-conus-v4"],
        "hrrr_configuration_cut": ["hrrr-conus-v4"],
        # No [static] table and a source whose metadata selects no file.
        "hrrr_v4_gsd41": [],
    }


def test_a_grid_at_another_spacing_does_not_ask_for_the_file():
    raw = _raw(ROOT / "configs" / "recipes" / "hrrr_configuration_cut.toml")
    raw["domain"][0]["dx"] = 1000.0
    assert required_static_rows(raw) == ()


def test_a_missing_static_file_is_fetched_once_with_its_size(tmp_path):
    raw = _raw(ROOT / "configs" / "recipes" / "hrrr_configuration_cut.toml")
    row = static_source_row("hrrr-conus-v4")
    lines, fetched = [], []

    def fetch(source_id, root, *, progress):
        fetched.append((source_id, root))
        progress("fetch-geog hrrr-conus-v4: 106 of 1,061 MiB (10%)")
        return root / "static_sources" / source_id / row.filename

    staged = stage_required_static_sources(raw, tmp_path, progress=lines.append,
                                           fetch=fetch)
    assert fetched == [("hrrr-conus-v4", tmp_path)]
    assert staged == [tmp_path / "static_sources" / "hrrr-conus-v4" / row.filename]
    assert lines[0].startswith("note: static source hrrr-conus-v4 is not staged; "
                               "fetching 1.11 GB from www.nco.ncep.noaa.gov into ")
    assert "gpuwm fetch-geog --static-source hrrr-conus-v4" in lines[0]
    assert lines[1] == "fetch-geog hrrr-conus-v4: 106 of 1,061 MiB (10%)"


def test_a_staged_static_file_is_verified_and_not_fetched(tmp_path, monkeypatch):
    raw = _raw(ROOT / "configs" / "recipes" / "hrrr_configuration_cut.toml")
    row = static_source_row("hrrr-conus-v4")
    present = tmp_path / "static_sources" / row.id / row.filename
    present.parent.mkdir(parents=True)
    present.write_bytes(b"x")
    monkeypatch.delenv(external_source.CACHE_ENV, raising=False)
    verified = []
    monkeypatch.setattr(external_source, "verify_local_file",
                        lambda path, r: verified.append(path))

    def fetch(*a, **k):
        raise AssertionError("a staged file was fetched again")

    lines = []
    assert stage_required_static_sources(raw, tmp_path, progress=lines.append,
                                         fetch=fetch) == [present]
    assert verified == [present]
    assert lines == []


def test_a_static_file_the_run_cannot_fetch_stops_the_chain_by_name(tmp_path, monkeypatch):
    from gpuwm.geog_assets import GeogFetchError

    def fail(*a, **k):
        raise GeogFetchError("static source 'hrrr-conus-v4' could not be fetched "
                             "from any of its 1 URL(s): HTTP 404")

    monkeypatch.setattr(external_source, "stage_required_static_sources",
                        stage_required_static_sources)
    monkeypatch.setattr(external_source, "fetch_static_source", fail)
    monkeypatch.delenv(external_source.CACHE_ENV, raising=False)
    raw = _raw(ROOT / "configs" / "recipes" / "hrrr_configuration_cut.toml")
    with pytest.raises(runplan.PlanError, match="could not be fetched"):
        runplan._stage_static_sources(raw, tmp_path, base_dir=tmp_path)


def test_the_dry_run_says_the_static_file_will_be_fetched(tmp_path, monkeypatch):
    monkeypatch.delenv(external_source.CACHE_ENV, raising=False)
    recipe = ROOT / "configs" / "recipes" / "hrrr_configuration_cut.toml"
    note = go_cli.static_source_note(_raw(recipe), tmp_path, base_dir=recipe.parent)
    assert note.startswith("go: static source hrrr-conus-v4 (1.11 GB) is not staged; "
                           "the run fetches it before the forcing")
    row = static_source_row("hrrr-conus-v4")
    present = tmp_path / "static_sources" / row.id / row.filename
    present.parent.mkdir(parents=True)
    present.write_bytes(b"x")
    assert go_cli.static_source_note(_raw(recipe), tmp_path,
                                     base_dir=recipe.parent) is None


class _Response(io.BytesIO):
    """An HTTP body served 100 bytes per read, as a slow link serves it."""

    status = 200
    headers = {}

    def read(self, size=-1):
        return super().read(100 if size is None or size < 0 else min(size, 100))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_a_large_download_says_its_progress(tmp_path):
    body = b"z" * 1000
    lines = []
    download_archive("https://example.invalid/file", tmp_path / "file.part",
                     expected_bytes=len(body), progress=lines.append, label="row",
                     urlopen_fn=lambda request: _Response(body),
                     report_every_bytes=100)
    steps = [line for line in lines if " of " in line]
    assert steps[0] == "fetch-geog row: 0 of 0 MiB (10%)"
    assert steps[-1].endswith("(100%)")
    assert len(steps) == 10


def test_a_download_with_no_reporting_asked_says_what_it_always_said(tmp_path):
    body = b"z" * 1000
    lines = []
    download_archive("https://example.invalid/file", tmp_path / "file.part",
                     expected_bytes=len(body), progress=lines.append, label="row",
                     urlopen_fn=lambda request: _Response(body))
    assert not [line for line in lines if " of " in line]


# ---------------------------------------------------------------------------
# The front door itself, on every shipped recipe, with no card.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("recipe", RECIPES, ids=lambda p: p.stem)
def test_gpuwm_go_plans_every_shipped_recipe_from_its_installed_path(
        recipe, tmp_path, monkeypatch, capsys):
    """`gpuwm go RECIPE --dry-run` resolves the whole chain for every recipe.

    The recipe is named where it ships, with nothing beside it but its
    siblings, which is exactly how the WOOF 1.0.3 smoke stopped before any
    stage.  The dry run walks the same door the run does (the route
    resolution that renders the namelist, the memory and plan checks) and
    stops before the fetch, so no card and no network are used.
    """
    from gpuwm import cli

    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
    monkeypatch.delenv(external_source.CACHE_ENV, raising=False)
    geog = tmp_path / "geog"
    geog.mkdir()
    code = cli.main(["go", str(recipe), "--dry-run", "--outdir", str(tmp_path / "out"),
                     "--geog-root", str(geog)])
    said = capsys.readouterr()
    assert code == 0, said.err[-2000:]
    assert "namelist.wps" not in said.err
    reads_static = bool(required_static_rows(_raw(recipe), base_dir=recipe.parent))
    assert ("static source hrrr-conus-v4 (1.11 GB) is not staged" in said.out) is reads_static
    # Nothing was written beside the shipped recipe.
    assert not list(recipe.parent.glob("*.namelist.wps"))


def test_the_staged_plan_auto_stages_the_static_file_before_the_forcing(
        tmp_path, monkeypatch, capsys):
    """Through the chain's own staging, with fetch-geog's fetch observed.

    The shipped cut, no static file under the GEOG root: the chain calls
    fetch_static_source for hrrr-conus-v4 into that root, says the size
    first, and only then reaches its forcing fetch.
    """
    config = tmp_path / "hrrr_configuration_cut.toml"
    shutil.copy(ROOT / "configs" / "recipes" / "hrrr_configuration_cut.toml", config)
    exp = load_experiment(config)
    geog = tmp_path / "geog"
    order = []

    class Stop(Exception):
        pass

    def fetch_static(source_id, root, *, progress):
        order.append(("static", source_id, Path(root)))
        progress("fetch-geog hrrr-conus-v4: 1,061 of 1,061 MiB (100%)")
        return Path(root) / "static_sources" / source_id / "hrrr_geo_em.d01.nc"

    def fetch_forcing(*a, **k):
        order.append(("forcing",))
        raise Stop

    monkeypatch.delenv(external_source.CACHE_ENV, raising=False)
    # conftest pins the chains' staging off the network; this cell is the
    # one that drives the real staging through the chain.
    monkeypatch.setattr(external_source, "stage_required_static_sources",
                        stage_required_static_sources)
    monkeypatch.setattr(external_source, "fetch_static_source", fetch_static)
    monkeypatch.setattr(runplan, "_run_fetch", fetch_forcing)
    from types import SimpleNamespace

    plan = SimpleNamespace(run_options={"data_dir": str(tmp_path / "data"),
                                        "geog_root": str(geog)}, config_intent={})
    observer = SimpleNamespace(enter_stage=lambda *a, **k: None,
                               finish_stage=lambda **k: None,
                               warn=lambda *a, **k: None,
                               arm_first_products=lambda *a, **k: None)
    with pytest.raises(Stop):
        runplan._staged_chain(plan, config_path=config, exp=exp,
                              observer=observer, run_dir=tmp_path / "run")
    assert order == [("static", "hrrr-conus-v4", geog), ("forcing",)]
    err = capsys.readouterr().err
    assert "note: static source hrrr-conus-v4 is not staged; fetching 1.11 GB" in err
    assert "note: fetch-geog hrrr-conus-v4: 1,061 of 1,061 MiB (100%)" in err
