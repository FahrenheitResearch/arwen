"""The HRRR route's emitted namelists carry every run field of the TOML.

The route prepares from the namelist pair it writes and the mirrored WRF
arm runs it, and its importer fills every setting the recipe table
declares for ``hrrr`` wherever that pair is silent.  An emission that
left a TOML value unwritten was therefore read back as the recipe's
value (a generic RUC snow form read as the fork's, the monthly surface
fields read as on), and the round trip did not notice, because it
compared only the prepared-cache identity, which drops every
preparation-inert field.  These tests pin both halves of the fix: each
such value is now written and reads back as the TOML says, and the round
trip compares every RunConfig field the pair carries or should carry and
refuses a drift by name.  The fields the forecast reads from the TOML
alone, which no namelist can spell, are exempt with their reader named
(``ROUTE_FORECAST_TOML_FIELDS``), and every shipped HRRR-route config
still emits its pair.
"""
from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
import tomllib

import pytest

from gpuwm.experiment import build_experiment, load_experiment
from gpuwm.hrrr_route_inputs import (
    DIFF6_FACTOR2_FORK_DEFAULT, ROUTE_FORECAST_TOML_FIELDS,
    HrrrRouteInputError, route_input_paths, verify_round_trip,
    write_hrrr_route_inputs)
from gpuwm.namelist_import import import_namelists, parse_namelist
from gpuwm.physics_source_defaults import (
    read_physics_selector_comment, recipe_physics_defaults)

ROOT = Path(__file__).parents[1]
RECIPE = ROOT / "configs/recipes/hrrr_configuration_clock.toml"


def _emit(tmp_path, name, **settings):
    """Emit the route inputs for the shipped HRRR recipe with ``settings``."""

    from gpuwm.companion_domains import candidate_wps_text

    exp = load_experiment(RECIPE)
    exp = replace(exp, domains=tuple(
        replace(domain, run=replace(domain.run, **settings))
        for domain in exp.domains))
    raw = tomllib.loads(RECIPE.read_text(encoding="utf-8"))
    raw["shared"].update({key: value for key, value in settings.items()
                          if value is not None})
    for key, value in settings.items():
        if value is None:
            raw["shared"].pop(key, None)
    output = tmp_path / f"{name}.toml"
    wps = candidate_wps_text(raw, exp, exp, output)
    write_hrrr_route_inputs(
        output, exp, wps_text=wps,
        writer=lambda path, text: path.write_text(text, encoding="utf-8"))
    return exp, route_input_paths(output)


def _replay(paths):
    """The experiment the route reads from the emitted pair."""

    text, _ = import_namelists(
        paths["wps_namelist"], paths["namelist_input"], request_source="hrrr")
    return build_experiment(tomllib.loads(text), source="route replay")


#: One case per field the measured emissions drifted on, each at the
#: value a TOML can state that the recipe table fills differently.
DRIFTED = {
    "rdlai2d": {"rdlai2d": False},
    "usemonalb": {"usemonalb": False},
    "ruc_soilprop": {"ruc_soilprop": "wrf_461"},
    "ruc_irrigation": {"ruc_irrigation": "wrf_461"},
    "ruc_snow": {"ruc_snow": "wrf_461"},
    "ruc_qvg_cold_start": {"ruc_qvg_cold_start": "wrf"},
    "ruc_2m_diagnostic": {"ruc_2m_diagnostic": "flux"},
    "fractional_seaice": {"fractional_seaice": 1},
    "diff_6th_form": {"diff_6th_form": "wrf_461", "diff_6th_factor2": None},
    "upper_wind_limiter_form": {"upper_wind_limiter_form": "wrf_461"},
    "mp_zero_out": {"mp_zero_out": 0},
    "mp_zero_out_thresh": {"mp_zero_out_thresh": 1.0e-8},
    "mp_zero_out_all": {"mp_zero_out_all": 0},
    "alb_sol": {"alb_sol": 0},
    "mynn_sfclay_variant": {"mynn_sfclay_variant": "wrf_461"},
}


@pytest.mark.parametrize("field", sorted(DRIFTED))
def test_a_toml_value_the_recipe_fills_differently_reaches_the_route(
        tmp_path, field):
    settings = DRIFTED[field]
    exp, paths = _emit(tmp_path, field, **settings)
    replay = _replay(paths)
    for domain, replayed in zip(exp.domains, replay.domains):
        for key in settings:
            assert getattr(replayed.run, key) == getattr(domain.run, key), key
    # The fill these values defeat is real: without them the route reads
    # the recipe's value (fractional_seaice has no fill, only a row).
    recipe = recipe_physics_defaults("hrrr")
    if field != "fractional_seaice":
        assert recipe[field] != settings[field]


NOAH_DEMO = ROOT / "configs/hrrr_native_quick_demo.toml"


@pytest.mark.parametrize("field", (
    "rdlai2d", "usemonalb", "ruc_irrigation", "ruc_snow",
    "ruc_qvg_cold_start", "ruc_2m_diagnostic"))
def test_a_noah_tree_reads_back_its_own_surface_settings(tmp_path, field):
    """No fork RUC signature to lean on: the generic values are written.

    The shipped Noah demo states the generic monthly-off surface and the
    generic RUC forms; the recipe fill read it back as monthly-on and the
    fork forms, the measured drift in the shipped demos, the ensemble
    recipe door and the albedo front door.
    """

    from gpuwm.companion_domains import candidate_wps_text

    exp = load_experiment(NOAH_DEMO)
    assert exp.root.run.sf_surface_physics == 2
    raw = tomllib.loads(NOAH_DEMO.read_text(encoding="utf-8"))
    output = tmp_path / "noah.toml"
    write_hrrr_route_inputs(
        output, exp, wps_text=candidate_wps_text(raw, exp, exp, output),
        writer=lambda path, text: path.write_text(text, encoding="utf-8"))
    replay = _replay(route_input_paths(output))
    assert getattr(replay.root.run, field) == getattr(exp.root.run, field)
    assert getattr(exp.root.run, field) != recipe_physics_defaults("hrrr")[field]


def test_the_stock_twin_never_carries_the_native_only_albedo_zero(tmp_path):
    _, paths = _emit(tmp_path, "albsol-zero", alb_sol=0)
    native = parse_namelist(paths["namelist_input"])["physics"]
    stock = parse_namelist(paths["stock_namelist_input"])["physics"]
    assert native["alb_sol"] == [0]
    assert "alb_sol" not in stock


def test_an_implicit_fork_factor_is_the_fork_default_on_both_sides(tmp_path):
    exp, paths = _emit(tmp_path, "implicit-factor", diff_6th_factor2=None)
    assert exp.root.run.diff_6th_form == "noaa_wrf39"
    assert "diff_6th_factor2" not in parse_namelist(
        paths["namelist_input"])["dynamics"]
    assert _replay(paths).root.run.diff_6th_factor2 == \
        DIFF6_FACTOR2_FORK_DEFAULT


def test_the_restated_fork_factor_is_the_integrators_constant():
    """The comparison restates 0.04 to stay off the GPU import; bind it."""

    source = (ROOT / "gpuwm/core/dycore.py").read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if (isinstance(node, ast.Assign)
                and [target.id for target in node.targets
                     if isinstance(target, ast.Name)]
                == ["DIFF6_FACTOR2_FORK_DEFAULT"]):
            assert ast.literal_eval(node.value) == DIFF6_FACTOR2_FORK_DEFAULT
            break
    else:
        pytest.fail("gpuwm/core/dycore.py lost DIFF6_FACTOR2_FORK_DEFAULT")


def test_the_recipe_emission_keeps_its_comment_and_rows(tmp_path):
    """A TOML that states the recipe's own values writes no new carrier."""

    _, paths = _emit(tmp_path, "recipe")
    text = paths["namelist_input"].read_text(encoding="utf-8")
    carried = read_physics_selector_comment(text)
    assert "ruc_soilprop" not in carried
    physics = parse_namelist(paths["namelist_input"])["physics"]
    assert "fractional_seaice" not in physics
    assert physics["rdlai2d"] == [True] and physics["usemonalb"] == [True]


def test_a_drifted_preparation_inert_field_is_refused_by_name(tmp_path):
    """The widened round trip fires on a field the cache identity drops."""

    exp, paths = _emit(tmp_path, "drift", ruc_snow="wrf_461")
    namelist = paths["namelist_input"]
    text = namelist.read_text(encoding="utf-8")
    edited = text.replace('"ruc_snow":"wrf_461"', '"ruc_snow":"wrf_45"', 1)
    assert edited != text
    namelist.write_text(edited, encoding="utf-8")
    with pytest.raises(HrrrRouteInputError, match="ruc_snow"):
        verify_round_trip(exp, paths["wps_namelist"], namelist)


def test_a_field_the_forecast_reads_from_the_toml_emits(tmp_path):
    """Group (b): no spelling in the pair, read by the forecast, not refused.

    Round 1 of this fix refused this emission for a field the gsd_41 MYNN
    solver reads from the TOML the forecast is bound to; the value reaches
    the model without the pair, so the round trip must not compare it.
    """

    from gpuwm.hrrr_route_inputs import ROUTE_FORECAST_TOML_FIELDS

    assert "bl_mynn_gsd41_unsquared_qtke" in ROUTE_FORECAST_TOML_FIELDS
    exp, paths = _emit(tmp_path, "unsquared",
                       bl_mynn_gsd41_unsquared_qtke=True)
    assert exp.root.run.bl_mynn_gsd41_unsquared_qtke is True
    assert paths["namelist_input"].is_file()


def test_a_field_neither_the_pair_nor_the_forecast_table_carries_is_refused(
        tmp_path):
    """Groups (a) and (c) stay compared.

    moist_cq has no WRF key and is not read from the TOML alone (it is in
    the prepared identity), so a TOML value the pair cannot carry is
    still refused by name.
    """

    with pytest.raises(HrrrRouteInputError, match="moist_cq"):
        _emit(tmp_path, "moist-cq", moist_cq=False)


def _inert_run_fields():
    from gpuwm.ingest.prepared_cache import PREPARATION_INERT_RUN_FIELDS

    return {path.partition(".")[2] for path in PREPARATION_INERT_RUN_FIELDS
            if path.startswith("run.")}


@pytest.mark.parametrize("field", sorted(ROUTE_FORECAST_TOML_FIELDS))
def test_a_forecast_toml_field_is_inert_unspelled_and_read_where_cited(field):
    """What makes a group (b) exemption true, held per field.

    * preparation-inert: preparing from the pair's default stamps the
      same cache, so the forecast's own cache comparison cannot object;
    * no importer spelling: the pair cannot carry it at all;
    * the cited module names the field, so the reader is real.
    """

    from dataclasses import fields as dataclass_fields

    from gpuwm.config import RunConfig

    assert field in {item.name for item in dataclass_fields(RunConfig)}
    assert field in _inert_run_fields()
    importer = (ROOT / "gpuwm/namelist_import.py").read_text(encoding="utf-8")
    assert field not in importer
    reader = ROUTE_FORECAST_TOML_FIELDS[field].split()[0].rstrip(",")
    assert field in (ROOT / reader).read_text(encoding="utf-8"), reader


def test_every_compared_inert_field_has_a_spelling_in_the_pair():
    """No group (c) field among the ones only the widened trip compares.

    Every preparation-inert run field the round trip compares has a
    reader in the importer or the selector comment.  The breakage this
    names: an inert field with neither a spelling nor a group (b) entry is
    a TOML setting the pair silently drops, or one refused on this route
    for nothing the forecast lacks.  Add a spelling (renderer and
    importer) or a ROUTE_FORECAST_TOML_FIELDS entry naming its reader.
    """

    from gpuwm.hrrr_route_inputs import round_trip_exempt_run_fields
    from gpuwm.physics_source_defaults import PHYSICS_SELECTOR_VALUES

    importer = (ROOT / "gpuwm/namelist_import.py").read_text(encoding="utf-8")
    unspelled = sorted(
        field for field in _inert_run_fields() - round_trip_exempt_run_fields()
        if field not in PHYSICS_SELECTOR_VALUES
        and f'"{field}' not in importer and f"{field} =" not in importer)
    assert unspelled == []


def _routed_configs():
    """Shipped configs whose own [fetch] source dispatches to this route.

    configs/frozen/ is left out: it is an archive of the bytes committed
    receipts ran under, not a menu (its README).
    """

    from gpuwm.source_drivability import candidate_route_chain

    routed = []
    for path in sorted((ROOT / "configs").rglob("*.toml")):
        if "frozen" in path.parts:
            continue
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        if "experiment" not in raw:
            continue
        source = (raw.get("fetch") or {}).get("source")
        if candidate_route_chain(source) == "prepared:hrrr":
            routed.append(path.relative_to(ROOT).as_posix())
    return routed


#: moist_cq = false is a verification counterfactual these namelists
#: cannot encode (ROUTE_IMPLICIT_SWITCHES).  These two attempts have been
#: refused for it since before this change; the refusal names it alone.
MOIST_CQ_COUNTERFACTUALS = {
    "configs/les_tornado_100m_mayfield_20211210_attempt2.toml",
    "configs/les_tornado_100m_mayfield_20211210_attempt2b.toml",
}

TORNADO_LES = tuple(
    f"configs/les_tornado_100m_{name}.toml" for name in (
        "dodgecity_20160524", "mayfield_20211210",
        "mayfield_20211210_attempt3", "mayfield_20211210_attempt3_fine30s"))


@pytest.mark.parametrize("config", _routed_configs())
def test_every_shipped_hrrr_route_config_emits_its_pair(tmp_path, config):
    """Every routed shipped config writes its pair, the four tornado LES
    configs round 1 refused (inflow_perturbation on d03) among them."""

    import re

    from gpuwm.companion_domains import candidate_wps_text

    path = ROOT / config
    exp = load_experiment(path)
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    beside = route_input_paths(path)
    wps = (beside["wps_namelist"].read_text(encoding="utf-8")
           if beside["wps_namelist"].is_file()
           else candidate_wps_text(raw, exp, exp, path))
    output = tmp_path / path.name

    def emit():
        write_hrrr_route_inputs(
            output, exp, wps_text=wps,
            writer=lambda target, text: target.write_text(
                text, encoding="utf-8"))

    if config in MOIST_CQ_COUNTERFACTUALS:
        with pytest.raises(HrrrRouteInputError) as refusal:
            emit()
        named = set(re.findall(r"d\d\d (\w+): config", str(refusal.value)))
        assert named == {"moist_cq"}
        return
    emit()


@pytest.mark.parametrize("config", TORNADO_LES)
def test_the_tornado_les_configs_are_routed_and_perturb_their_inflow(config):
    exp = load_experiment(ROOT / config)
    assert any(domain.run.inflow_perturbation for domain in exp.domains)
    assert config in _routed_configs()


def test_slope_radiation_and_terrain_shading_reach_both_halves(tmp_path):
    """WRF keys the importer reads and the renderer never wrote: a TOML
    slope_rad = 1 emitted a pair that said 0 (and, in round 1, refused)."""

    exp, paths = _emit(tmp_path, "slope", slope_rad=1, topo_shading=1,
                       shadlen=12500.0)
    replay = _replay(paths)
    for domain, replayed in zip(exp.domains, replay.domains):
        for key in ("slope_rad", "topo_shading", "shadlen"):
            assert getattr(replayed.run, key) == getattr(domain.run, key)
    native = parse_namelist(paths["namelist_input"])["physics"]
    stock = parse_namelist(paths["stock_namelist_input"])["physics"]
    for key in ("slope_rad", "topo_shading", "shadlen"):
        assert native[key] == stock[key]


def test_noah_mosaic_reaches_both_halves(tmp_path):
    from gpuwm.companion_domains import candidate_wps_text

    exp = load_experiment(NOAH_DEMO)
    exp = replace(exp, domains=tuple(
        replace(domain, run=replace(domain.run, sf_surface_mosaic=1,
                                    mosaic_cat=5))
        for domain in exp.domains))
    raw = tomllib.loads(NOAH_DEMO.read_text(encoding="utf-8"))
    output = tmp_path / "mosaic.toml"
    write_hrrr_route_inputs(
        output, exp, wps_text=candidate_wps_text(raw, exp, exp, output),
        writer=lambda path, text: path.write_text(text, encoding="utf-8"))
    paths = route_input_paths(output)
    replay = _replay(paths)
    assert replay.root.run.sf_surface_mosaic == 1
    assert replay.root.run.mosaic_cat == 5
    native = parse_namelist(paths["namelist_input"])["physics"]
    stock = parse_namelist(paths["stock_namelist_input"])["physics"]
    assert native["mosaic_cat"] == stock["mosaic_cat"] == [5]


def test_a_mosaic_tile_count_beside_mosaic_off_is_dead_state(tmp_path):
    """mosaic_cat is unread with mosaic off (the experiment fingerprint
    drops it), so it is not a drift and nothing is written for it."""

    _, paths = _emit(tmp_path, "mosaic-off", mosaic_cat=5)
    assert "mosaic_cat" not in parse_namelist(
        paths["namelist_input"])["physics"]
