"""WRF's ``cycling`` reaches both halves of the HRRR route's namelist pair.

Acceptance defect D-04 (2.8.8 matrix at c8f95278882b): the importer maps
``&time_control cycling = .true.`` onto ``[shared] cycling`` (every
operational HRRR namelist states it), but the HRRR route's namelist writer
never emitted the key.  The route's round trip re-imports what it wrote,
read ``cycling`` back as WRF's default ``.false.``, and refused every
imported operational HRRR namelist through the site preparer with
``d01 cycling: config True vs namelist False``.  The writer now carries
the key on both halves, so the preparation, the mirrored WRF arm and the
forecast all run the cycled start the TOML states.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tomllib

import pytest

from gpuwm.experiment import build_experiment, load_experiment
from gpuwm.hrrr_route_inputs import (
    HrrrRouteInputError, render_namelist_input, route_input_paths,
    write_hrrr_route_inputs)
from gpuwm.namelist_import import import_namelists, parse_namelist

ROOT = Path(__file__).parents[1]
RECIPE = ROOT / "configs/recipes/hrrr_configuration_clock.toml"
OPERATIONAL = ROOT / "tests/data/hrrr_v4121_namelists"


def _write(path, text):
    path.write_text(text, encoding="utf-8")


def _with_cycling(exp, value):
    return replace(exp, domains=tuple(
        replace(domain, run=replace(domain.run, cycling=value))
        for domain in exp.domains))


def _emit(tmp_path, exp, raw, name):
    from gpuwm.companion_domains import candidate_wps_text

    output = tmp_path / f"{name}.toml"
    write_hrrr_route_inputs(output, exp,
                            wps_text=candidate_wps_text(raw, exp, exp, output),
                            writer=_write)
    return route_input_paths(output)


def _replay(paths):
    text, _ = import_namelists(
        paths["wps_namelist"], paths["namelist_input"], request_source="hrrr")
    return build_experiment(tomllib.loads(text), source="route replay")


def test_a_cycled_configuration_emits_and_reads_back_cycled(tmp_path):
    """The refusal D-04 reproduced, on the shipped recipe with cycling on."""

    exp = _with_cycling(load_experiment(RECIPE), True)
    raw = tomllib.loads(RECIPE.read_text(encoding="utf-8"))
    raw["shared"]["cycling"] = True
    paths = _emit(tmp_path, exp, raw, "cycled")
    assert all(domain.run.cycling is True
               for domain in _replay(paths).domains)
    for half in ("namelist_input", "stock_namelist_input"):
        parsed = parse_namelist(paths[half])["time_control"]["cycling"]
        assert parsed in (True, [True]), half


def test_the_default_emission_writes_no_cycling_row():
    """WRF's default is .false.; an uncycled emission keeps its bytes."""

    exp = load_experiment(RECIPE)
    assert exp.root.run.cycling is False
    for stock in (False, True):
        assert "cycling" not in render_namelist_input(exp, stock=stock)


def test_cycling_is_one_value_for_the_whole_run():
    """WRF's Registry declares cycling a scalar: a split tree is refused."""

    exp = load_experiment(ROOT / "configs/real74_4dom.toml")
    assert len(exp.domains) == 4
    domains = list(exp.domains)
    domains[-1] = replace(domains[-1],
                          run=replace(domains[-1].run, cycling=True))
    with pytest.raises(HrrrRouteInputError, match="cycling differs"):
        render_namelist_input(replace(exp, domains=tuple(domains)))


def test_the_imported_operational_hrrr_namelist_prepares_on_this_route(
        tmp_path):
    """The acceptance row's reproduction, without the site.

    C1a and N1.1 imported the vendored operational HRRR v4.1.21 pair
    (times set, the static datasets the door admits) and handed the TOML
    to the route's preparation, which writes this pair and round-trips it.
    """

    wps = (OPERATIONAL / "hrrr_namelist.wps").read_text(encoding="utf-8")
    wps = wps.replace("2009-08-21_00:00:00", "2018-08-26_12:00:00")
    wps = wps.replace("2009-08-21_12:00:00", "2018-08-26_18:00:00")
    wps = wps.replace("'modis_15s+modis_fpar+modis_lai+30s'",
                      "'modis_lai+30s'")
    (tmp_path / "namelist.wps").write_text(wps, encoding="utf-8")
    text, _ = import_namelists(tmp_path / "namelist.wps",
                               OPERATIONAL / "hrrr_wrf.nl", name="d04")
    config = tmp_path / "d04.toml"
    config.write_text(text, encoding="utf-8")
    exp = build_experiment(tomllib.loads(text), source=str(config))
    assert exp.root.run.cycling is True
    write_hrrr_route_inputs(config, exp, wps_text=wps, writer=_write)
    paths = route_input_paths(config)
    assert _replay(paths).root.run.cycling is True
