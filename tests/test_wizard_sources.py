"""``gpuwm domain --source`` is the source registry, not a hand-held list.

The 2026-08-17 model battery ran ten init sources end to end and found the
wizard door open for three of them: ``gpuwm domain --source`` accepted
``{gfs, hrrr, era5}``, so every other battery TOML was hand-assembled from a
``--root-dx 3`` emission with the boundary cadence typed in by a person.  A
source with a runnable profile and no way to plan a domain for it is
engine-proven, not shipped.

These tests bind the closure, and they are deliberately written so that not
one of them names a model in an assertion about mechanism:

* every registered runnable source plans, and its emitted ``namelist.wps``
  carries the cadence its REGISTRY ROW declares;
* a regional source refuses an out-of-coverage plan AT PLAN TIME, naming the
  corner, the source index it maps to and the window -- the ICON-EU failure
  the battery could only get out of a preparation traceback;
* a row nobody has written code for still plans (the arbitrary acceptance
  test: a synthetic registry row goes through the whole door);
* a registered row with no runnable route refuses by name rather than by
  argparse's "invalid choice".
"""
from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from gpuwm.cli import main as cli_main
from gpuwm import source_adapters as registry
from gpuwm.source_adapters import (AdapterStatus, LambertGridWindow,
                                   RegularLatLonWindow, get_source_adapter,
                                   source_coverage_window,
                                   source_forcing_interval_seconds,
                                   wizard_planable_source_ids)
from gpuwm.source_authorities import packaged_authorities
from gpuwm.source_coverage import window_centre

#: A point every global source reaches, in the middle of the certified
#: CONUS envelope so the regional rows that also cover it plan there too.
_GLOBAL_POINT = (38.5, -97.5)


def _plan_point(source: str) -> tuple[float, float]:
    """A point inside SOURCE's own coverage, taken from its registry row.

    Table-driven on purpose: a per-source point table in a test is the same
    per-model bandaid the registry exists to abolish, and it would pass
    while the door stayed shut for a model nobody remembered to list.
    """

    centre = window_centre(source_coverage_window(source))
    return _GLOBAL_POINT if centre is None else centre


def _emit(tmp_path: Path, source: str, *extra: str,
          point: tuple[float, float] | None = None,
          cycle: str = "2026-08-17T00", hours: str = "6",
          extra_card: str = "16gb") -> tuple[int, Path]:
    lat, lon = point if point is not None else _plan_point(source)
    out = tmp_path / f"{source.replace('-', '_')}.toml"
    rc = cli_main([
        "domain", f"--point={lat:.4f},{lon:.4f}", "--card", extra_card,
        "--root-dx", "3", "--hours", hours, "--source", source,
        "--cycle", cycle, "--out", str(out), *extra])
    return rc, out


# ---------------------------------------------------------------------------
# The door: every registered runnable source plans.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("source", wizard_planable_source_ids())
def test_every_registered_source_plans_a_runnable_config(tmp_path, source):
    rc, out = _emit(tmp_path / source, source)
    assert rc == 0, f"{source} did not plan"
    assert out.is_file()
    config = tomllib.loads(out.read_text(encoding="utf-8"))
    assert config["experiment"]["run_seconds"] == 21600.0
    # HRRR writes its route inputs instead of the bare companion namelist;
    # both spellings put a namelist.wps beside the config.
    namelists = sorted(out.parent.glob("*.namelist.wps"))
    assert namelists, f"{source} emitted no namelist.wps"
    declared = int(source_forcing_interval_seconds(source))
    text = namelists[0].read_text(encoding="utf-8")
    assert f"interval_seconds = {declared},", (
        f"{source}: namelist.wps must carry the registry's cadence")
    assert f" interval_seconds = {declared},\n" in text


@pytest.mark.parametrize("source", wizard_planable_source_ids())
def test_emitted_config_reloads_through_the_real_loader(tmp_path, source):
    from gpuwm.domain_wizard import experiment_from_text

    rc, out = _emit(tmp_path / source, source)
    assert rc == 0
    exp = experiment_from_text(out.read_text(encoding="utf-8"),
                              source=str(out))
    assert exp.run_seconds == 21600.0


def _sources_with_public_bytes() -> frozenset[str]:
    """Every source `gpuwm fetch` can download, read from the AUTHORITY.

    Deliberately not ``gpuwm.fetch.fetch_front_door_sources()``: that is
    the seam under test here, and a test that asks the seam what the seam
    should say passes while the door is shut.  The two things a fetch can
    run are a row in the packaged acquisition-route document and one of
    the hand-written transports that predate it, so this reads the JSON
    itself and adds the named legacy list.  The 2026-08-17 defect this
    pins: ten routes were open and the hint seam still spelled four
    models by hand, so `gpuwm domain` printed "stage the bytes yourself"
    for every model whose route had just landed.
    """

    from gpuwm.fetch_routes import LEGACY_ROUTE_SOURCES, ROUTE_TABLE_NAME

    table = json.loads(
        (Path(registry.__file__).with_name("authorities") / ROUTE_TABLE_NAME)
        .read_text(encoding="utf-8"))
    return frozenset(table["routes"]) | frozenset(LEGACY_ROUTE_SOURCES)


def _routed_sources() -> frozenset[str]:
    """The table-driven half, read from the authority document."""

    from gpuwm.fetch_routes import ROUTE_TABLE_NAME

    table = json.loads(
        (Path(registry.__file__).with_name("authorities") / ROUTE_TABLE_NAME)
        .read_text(encoding="utf-8"))
    return frozenset(table["routes"])


_ROUTED_SOURCES = _routed_sources()


def _refused_source_ids() -> frozenset[str]:
    """The runnable rows the route authority refuses by name."""

    from gpuwm.fetch_routes import ROUTE_TABLE_NAME

    table = json.loads(
        (Path(registry.__file__).with_name("authorities") / ROUTE_TABLE_NAME)
        .read_text(encoding="utf-8"))
    return frozenset(table["refusals"])


@pytest.mark.parametrize("source", wizard_planable_source_ids())
def test_fetch_table_is_emitted_exactly_where_the_fetch_door_reaches(
        tmp_path, source):
    """A ``[fetch]`` table is a claim that `gpuwm fetch` can get the bytes.

    Emitting one for a source the fetch door does not serve produces a
    config that is refused at every later load; omitting one for a source
    it DOES serve loses the download hint.  Both are decided against the
    route authority, never against the emitting seam and never against a
    list written here.
    """

    rc, out = _emit(tmp_path / source, source)
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    config = tomllib.loads(text)
    expected = source in _sources_with_public_bytes()
    assert ("fetch" in config) is expected, source
    if not expected:
        # And the gap is stated in the file, not merely left blank.
        assert "NO [fetch] TABLE" in text


@pytest.mark.parametrize("source", sorted(_sources_with_public_bytes()))
def test_routed_source_prints_a_runnable_fetch_next_step(
        tmp_path, capsys, source):
    """Step 1 of the printed recipe is a command, not an apology.

    A source whose bytes this ArWen can download must get a pasteable
    ``gpuwm fetch`` line out of `gpuwm domain`; printing "stage the bytes
    yourself" for a model with a live route is the field exhibit that
    makes a reader conclude the route does not exist.
    """

    rc, out = _emit(tmp_path / source, source)
    assert rc == 0
    printed = capsys.readouterr().out
    assert f"gpuwm fetch --source {source} " in printed, printed
    assert "has no download route" not in printed, printed
    config = tomllib.loads(out.read_text(encoding="utf-8"))
    assert config["fetch"]["source"] == source


@pytest.mark.parametrize("source", sorted(_ROUTED_SOURCES))
def test_printed_fetch_step_plans_through_the_real_planner(
        tmp_path, capsys, source):
    """Step 1 is RUN, not read: its argv goes through the real planner.

    ``fetch_routes.resolve_request`` decides everything a fetch will do
    before a byte moves, so planning the printed command proves the step
    exits 0 rather than 2.  The defect class: the wizard printed
    ``--area`` at every source, and a table route publishes whole objects
    with no subsetting service in front of them, so the emitted recipe's
    first line was refused for every model with a route.
    """

    import shlex

    from gpuwm.fetch import parse_cycle
    from gpuwm import fetch_routes

    rc, _ = _emit(tmp_path / source, source)
    assert rc == 0
    line = next(l for l in capsys.readouterr().out.splitlines()
                if "gpuwm fetch --source " in l)
    argv = shlex.split(line.strip(), posix=False)
    flags = {argv[i]: argv[i + 1] for i in range(len(argv) - 1)
             if argv[i].startswith("--")}
    fetch_routes.resolve_request(
        source, cycle=parse_cycle(flags["--cycle"], source),
        hours=int(flags["--hours"]),
        cadence=(int(flags["--cadence"]) if "--cadence" in flags else None),
        start_hour=int(flags.get("--forecast-start-hour", 0)),
        host=None, member=None,
        area=flags.get("--area"), out=Path(flags["--out"]))


@pytest.mark.parametrize("source", sorted(_sources_with_public_bytes()))
def test_hand_written_fetch_table_validates_for_every_routed_source(source):
    """A hand-written ``[fetch]`` table naming a routed source loads.

    The validator and the emitter read one definition of "can be
    fetched", so a table typed by hand for any downloadable source is
    accepted rather than refused as an unknown source.
    """

    from gpuwm.fetch import validate_fetch_hints

    validate_fetch_hints({"source": source, "cycle": "2026-08-17T00",
                          "hours": 6, "out": "data"},
                         source="hand-written.toml")


# ---------------------------------------------------------------------------
# Regional coverage: refused at plan time, with the breakage named.
# ---------------------------------------------------------------------------

def _regional_sources() -> tuple[str, ...]:
    return tuple(s for s in wizard_planable_source_ids()
                 if source_coverage_window(s) is not None)


@pytest.mark.parametrize("source", _regional_sources())
def test_regional_source_plans_inside_its_own_window(tmp_path, source):
    rc, out = _emit(tmp_path / source, source)
    assert rc == 0
    assert out.is_file()


@pytest.mark.parametrize("source", _regional_sources())
def test_regional_source_refuses_out_of_coverage_at_plan_time(
        tmp_path, capsys, source):
    """Half a world east of the grid centre, at the same latitude.

    Same hemisphere, same projection family, so what refuses is coverage
    and nothing else: every shipped regional window spans well under 180
    degrees of longitude, so this point is outside all of them.
    """

    lat, lon = _plan_point(source)
    opposite = (lat, ((lon + 180.0 + 180.0) % 360.0) - 180.0)
    rc, out = _emit(tmp_path / source, source, point=opposite)
    err = capsys.readouterr().err
    assert rc == 2, err
    assert not out.exists(), "a refused plan must write nothing"
    assert "Traceback" not in err
    assert source in err or source.upper() in err
    # The substance the battery's preparation-stage message got right and
    # the wizard could not say at all: where the target lands in the
    # SOURCE's own index space, and what the source covers.  HRRR's
    # certified route says it in its own words (it knows about the
    # interpolation halo as well as the rectangle); every other regional
    # source says it in the generic window's words.  Both name indices.
    assert any(token in err for token in ("source index", "i=", "j=")), err
    assert "choose a source" in err, err


#: What the 2026-08-17 battery's ICON-EU PREPARATION printed, on real
#: bytes, after decoding 1,752 objects and 73 seconds of work:
#:
#:   target point (0, 0) at lat/lon (34.2972, -102.4465) maps to source
#:   index x=-1263.144 y=76.756, and the source covers x=0..1376
#:   (lon -23.5..62.5) y=0..656 (lat 29.5..70.5)
#:
#: The wizard now answers the same question from the registry before
#: anything is downloaded, so the two answers have to agree.
_BATTERY_ICON_EU_PREP_INDEX = (-1263.144, 76.756)
_BATTERY_DOMAIN = dict(point=(38.5, -97.5), nx=300, ny=300, dx_m=3000.0)


def test_plan_time_refusal_agrees_with_the_measured_preparation_refusal():
    """Same corner, same source index, from table data instead of bytes."""

    import re

    from gpuwm.domain_wizard import (_projection_entries,
                                     source_coverage_refusal)

    lat, lon = _BATTERY_DOMAIN["point"]
    message = source_coverage_refusal(
        _projection_entries(lat, lon, "auto"),
        _BATTERY_DOMAIN["nx"], _BATTERY_DOMAIN["ny"],
        source="icon-eu", root_dx_m=_BATTERY_DOMAIN["dx_m"])
    assert message is not None
    found = re.search(r"x=(-?\d+\.\d+) y=(-?\d+\.\d+)", message)
    assert found, message
    planned = (float(found.group(1)), float(found.group(2)))
    # ICON-EU's grid step is 0.0625 deg; the preparation reports the
    # target's own corner and the wizard the root's corner mass point, so
    # they are one half-cell apart by construction and no further.
    for planned_value, measured in zip(planned, _BATTERY_ICON_EU_PREP_INDEX):
        assert abs(planned_value - measured) < 1.0, message
    # And the window itself is quoted exactly as the preparation quoted it.
    assert "x=0..1376" in message and "y=0..656" in message


def test_a_domain_near_a_regional_edge_warns_instead_of_refusing(
        tmp_path, capsys):
    """Warn where the MARGIN overruns; refuse only where the DOMAIN does.

    A fitted root inside the grid whose margined forcing box runs off the
    north edge is a legal plan -- for a coverage-boxed source ``--area``
    is a coverage check, not a crop -- so the box is clamped into the
    grid and the clamp is disclosed.  This path existed for HRRR alone;
    every regional row reaches it now.
    """

    rc, out = _emit(tmp_path, "rrfs", point=(51.0, -100.0), extra_card="24gb")
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert out.is_file()
    assert "clamped" in captured.err
    assert "RRFS coverage ends there" in captured.err


def test_out_of_coverage_refusal_names_the_offending_corner(tmp_path, capsys):
    """The measured battery case, at plan time instead of at prep time.

    ICON-EU over the battery's central-US domain: the run cost a full
    acquisition and 73 s of preparation to learn the grid does not reach
    the target.  Same answer, before anything is downloaded.
    """

    rc, out = _emit(tmp_path, "icon-eu", point=(38.5, -97.5))
    err = capsys.readouterr().err
    assert rc == 2
    assert not out.exists()
    assert "icon-eu" in err
    assert "29.5" in err and "70.5" in err and "-23.5" in err
    assert "62.5" in err


# ---------------------------------------------------------------------------
# The arbitrary acceptance test: a row nobody wrote code for.
# ---------------------------------------------------------------------------

def _install_row(monkeypatch, adapter) -> None:
    adapters = registry._ADAPTERS + (adapter,)
    aliases = dict(registry._ALIASES)
    for name in (adapter.source_id, *adapter.aliases):
        aliases[name.strip().lower().replace("_", "-")] = adapter
    monkeypatch.setattr(registry, "_ADAPTERS", adapters)
    monkeypatch.setattr(registry, "_ALIASES", aliases)


def _synthetic(source_id: str, **overrides):
    fields = dict(
        source_id=source_id, aliases=(), upstream_model_id=None,
        source_kind=registry.SourceKind.DETERMINISTIC_STATE,
        file_family="GRIB2", decoder="packaged profile + GRIB2 bridges",
        default_product="pres", required_products=("pres",),
        max_forecast_hour=48,
        upstream_ingest="declarative_mapping_v1_over_packaged_profile",
        status=AdapterStatus.RUNNABLE_NOT_CERTIFIED,
        field_mapping="probe", level_mapping="probe",
        cadence_mapping="probe", stock_wrf_gate="probe",
        runnable=True, runner="mapped_composition_v1",
        forcing_interval_seconds=7200.0,
        notes="synthetic registry row used to prove zero code per model",
    )
    fields.update(overrides)
    return registry.SourceAdapter(**fields)


def test_a_new_registry_row_plans_with_no_new_code(tmp_path, monkeypatch):
    """Adding a model must be a ROW.  This adds one and uses the door."""

    _install_row(monkeypatch, _synthetic("probe-global"))
    rc, out = _emit(tmp_path, "probe-global", point=_GLOBAL_POINT)
    assert rc == 0
    namelist = out.parent / f"{out.stem}.namelist.wps"
    # The cadence the ROW declared, in the emitted namelist, with nothing
    # in gpuwm/domain_wizard.py knowing this source exists.
    assert " interval_seconds = 7200,\n" in namelist.read_text(
        encoding="utf-8")


def test_a_new_regional_row_refuses_out_of_its_declared_window(
        tmp_path, monkeypatch, capsys):
    window = RegularLatLonWindow(south=-40.0, west=110.0,
                                 north=-10.0, east=155.0, nx=721, ny=481)
    _install_row(monkeypatch,
                 _synthetic("probe-regional", coverage_window=window))
    inside = window_centre(window)
    rc, out = _emit(tmp_path / "in", "probe-regional", point=inside)
    assert rc == 0, capsys.readouterr().err
    rc, out = _emit(tmp_path / "out", "probe-regional",
                    point=_GLOBAL_POINT)
    err = capsys.readouterr().err
    assert rc == 2
    assert not out.exists()
    assert "probe-regional" in err and "110" in err and "155" in err


def test_a_row_with_no_cadence_refuses_by_naming_the_missing_fact(
        tmp_path, monkeypatch, capsys):
    _install_row(monkeypatch,
                 _synthetic("probe-cadenceless",
                            forcing_interval_seconds=None))
    rc, out = _emit(tmp_path, "probe-cadenceless", point=_GLOBAL_POINT)
    err = capsys.readouterr().err
    assert rc == 2
    assert not out.exists()
    assert "cadence" in err


# ---------------------------------------------------------------------------
# Registered but not plannable: refused by name, never by "invalid choice".
# ---------------------------------------------------------------------------

def _unplannable() -> tuple[str, ...]:
    plannable = set(wizard_planable_source_ids())
    return tuple(a.source_id for a in registry.source_adapters()
                 if a.source_id not in plannable)


@pytest.mark.parametrize("source", _unplannable())
def test_registered_but_unrunnable_source_refuses_by_name(
        tmp_path, capsys, source):
    rc, out = _emit(tmp_path, source, point=_GLOBAL_POINT)
    err = capsys.readouterr().err
    assert rc == 2
    assert not out.exists()
    assert source in err
    assert "invalid choice" not in err
    assert "Traceback" not in err
    # A refusal stands only if it names the breakage: the reader is told
    # what is missing and which sources DO plan.
    assert any(word in err for word in ("no runnable", "cadence"))


def test_unknown_source_lists_the_registry(tmp_path, capsys):
    rc, out = _emit(tmp_path, "not-a-model", point=_GLOBAL_POINT)
    err = capsys.readouterr().err
    assert rc == 2
    assert not out.exists()
    assert "not-a-model" in err
    assert "gfs" in err and "era5" in err


def test_registry_alias_resolves_to_its_row(tmp_path):
    """`--source gem` was measured refusing while every doc said "gem"."""

    assert get_source_adapter("gem").source_id == "gem-gdps"
    rc, out = _emit(tmp_path, "gem", point=_GLOBAL_POINT)
    assert rc == 0
    config = tomllib.loads(out.read_text(encoding="utf-8"))
    assert config["experiment"]["name"]


# ---------------------------------------------------------------------------
# The declared facts are the SAME facts the decode route reads.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "source", [a.source_id for a in registry.source_adapters()
               if a.packaged_profile and a.forcing_interval_seconds])
def test_declared_cadence_matches_the_packaged_mapping(source):
    adapter = get_source_adapter(source)
    mapping = json.loads(
        packaged_authorities(adapter.packaged_profile)["mapping"]
        .read_text(encoding="utf-8"))
    assert (float(mapping["target"]["boundary_interval_seconds"])
            == adapter.forcing_interval_seconds), (
        f"{source}: the registry row and the packaged mapping disagree "
        "about the source's own boundary cadence")


def test_declared_conus_window_reproduces_the_native_grid_envelope():
    """The declared window is the grid, not a hand-held box.

    The retired CONUS box disagreed with the real grid by degrees; this
    binds the declaration to the computation the certified HRRR route
    already performs from the grid definition itself.
    """

    from gpuwm.ingest.hrrr_target import hrrr_coverage_envelope

    window = source_coverage_window("hrrr")
    assert isinstance(window, LambertGridWindow)
    assert window.envelope() == hrrr_coverage_envelope()
    # RRFS is HRRR's successor on HRRR's measured-identical grid.
    assert source_coverage_window("rrfs") is window


def test_fetch_and_wizard_read_one_coverage_definition():
    from gpuwm.fetch import source_coverage_envelope

    for source in wizard_planable_source_ids():
        window = source_coverage_window(source)
        expected = None if window is None else window.envelope()
        assert source_coverage_envelope(source) == expected, source


# ---------------------------------------------------------------------------
# The published page has to agree with the doors.
#
# docs/public/SOURCES.md said "`gpuwm fetch` downloads gfs, gdas, hrrr and
# era5" while docs/public/DATA.md published a working command for ten more.
# A reader who believes the first page hand-stages bytes they could have
# downloaded, so the contradiction is a defect in the product, not a typo.
# ---------------------------------------------------------------------------

def _sources_page() -> str:
    from pathlib import Path as _Path

    import gpuwm

    return (_Path(gpuwm.__file__).parent.parent
            / "docs" / "public" / "SOURCES.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("source", sorted(_sources_with_public_bytes()))
def test_the_sources_page_lists_every_source_the_fetch_door_serves(source):
    assert f"`{source}`" in _sources_page(), source


def test_the_sources_page_does_not_deny_a_route_that_exists():
    """No sentence claiming the fetch door reaches only the legacy four."""

    from gpuwm.fetch_routes import LEGACY_ROUTE_SOURCES

    page = _sources_page()
    legacy = ", ".join(f"`{name}`" for name in LEGACY_ROUTE_SOURCES[:-1])
    denial = f"downloads {legacy} and `{LEGACY_ROUTE_SOURCES[-1]}`"
    assert denial not in page, denial


@pytest.mark.parametrize("source", sorted(_refused_source_ids()))
def test_the_sources_page_names_every_refused_source(source):
    """The three runnable rows with no route are named, not omitted."""

    assert f"`{source}`" in _sources_page(), source
