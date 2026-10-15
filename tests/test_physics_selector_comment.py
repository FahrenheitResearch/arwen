"""Explicit generation metadata survives WRF-compatible route files."""
from pathlib import Path
import tomllib

import pytest

from gpuwm.experiment import build_experiment
from gpuwm.physics_source_defaults import (
    read_physics_selector_comment, with_physics_selector_comment)

FIXTURE = Path(__file__).parent / "fixtures/source_requests"
MARKER = "! gpuwm-physics-selectors-v1: "


def test_omitted_metadata_preserves_exact_namelist_bytes():
    text = (FIXTURE / "hrrr_wrf.nl.c18c").read_text()
    assert read_physics_selector_comment(text) == {}
    assert with_physics_selector_comment(text, {}) is text


@pytest.mark.parametrize("value", ("gsl_wrf39", "wrf_461"))
def test_explicit_metadata_survives_an_unnamed_import(tmp_path, value):
    from gpuwm.namelist_import import import_namelists, parse_namelist
    text = (FIXTURE / "hrrr_wrf.nl.c18c").read_text()
    carried = with_physics_selector_comment(text, {"mynn_sfclay_variant": value})
    assert with_physics_selector_comment(carried, {"mynn_sfclay_variant": value}) is carried
    original = tmp_path / "control.input"
    original.write_text(text)
    namelist = tmp_path / "namelist.input"
    namelist.write_text(carried)
    assert parse_namelist(namelist) == parse_namelist(original)
    resolved, _ = import_namelists(FIXTURE / "hrrr_namelist.wps.c18", namelist)
    assert build_experiment(tomllib.loads(resolved), source="carried selector").root.run.mynn_sfclay_variant == value


def test_explicit_legacy_metadata_overrides_a_named_source_default(tmp_path):
    from gpuwm.namelist_import import import_namelists
    namelist = tmp_path / "hrrr_wrf.nl"
    namelist.write_text(with_physics_selector_comment(
        (FIXTURE / "hrrr_wrf.nl.c18c").read_text(),
        {"mynn_sfclay_variant": "wrf_461"}))
    resolved, report = import_namelists(FIXTURE / "hrrr_namelist.wps.c18", namelist)
    assert build_experiment(tomllib.loads(resolved), source="explicit legacy").root.run.mynn_sfclay_variant == "wrf_461"
    assert any(row.key == "mynn_sfclay_variant" and row.value == "wrf_461"
               for row in report.defaults_applied)


@pytest.mark.parametrize("comment", (
    MARKER + '{"unknown": "gsl_wrf39"}',
    MARKER + '{"mynn_sfclay_variant": 5}',
    MARKER + '{"mynn_sfclay_variant": "unknown"}',
    MARKER + '["gsl_wrf39"]',
    MARKER + '{}',
    MARKER + '{"mynn_sfclay_variant": "wrf_461", "mynn_sfclay_variant": "gsl_wrf39"}',
    MARKER + '{',
    '! gpuwm-physics-selectors-v2: {}',
    MARKER + '{}\n' + MARKER + '{}',
))
def test_invalid_metadata_refuses(comment, tmp_path):
    from gpuwm.namelist_import import import_namelists
    with pytest.raises(ValueError, match="physics selector comment"):
        read_physics_selector_comment(comment)
    namelist = tmp_path / "namelist.input"
    namelist.write_text(comment + "\n" + (FIXTURE / "hrrr_wrf.nl.c18c").read_text())
    with pytest.raises(ValueError, match="physics selector comment"):
        import_namelists(FIXTURE / "hrrr_namelist.wps.c18", namelist)


@pytest.mark.parametrize("value", ("wrf_471", "wrf_480"))
def test_urban_water_balance_generation_survives_an_unnamed_import(tmp_path, value):
    """WHATS-NEW 2.9.0 names urban_water_balance = "wrf_471" as the way back
    to the 2.8.9 urban rain-evaporation rule. A stock WRF namelist has no
    spelling for it, so a route file carries it in the selector comment; the
    comment refused the key before it had a row, and the old rule could not
    ride a WRF route file."""
    from gpuwm.namelist_import import import_namelists
    text = (FIXTURE / "hrrr_wrf.nl.c18c").read_text()
    carried = with_physics_selector_comment(text, {"urban_water_balance": value})
    assert read_physics_selector_comment(carried) == {"urban_water_balance": value}
    namelist = tmp_path / "namelist.input"
    namelist.write_text(carried)
    resolved, _ = import_namelists(FIXTURE / "hrrr_namelist.wps.c18", namelist)
    run = build_experiment(tomllib.loads(resolved), source="carried selector").root.run
    assert run.urban_water_balance == value


def test_every_default_on_wrf_correction_has_a_route_carrier_for_its_old_generation():
    """The four WRF 4.7/4.8 corrections that change default output in 2.9.0
    each name an earlier generation users can select; each pair must be a
    selector-comment row, or the HRRR route file drops the user's choice."""
    from gpuwm.config import RunConfig
    from gpuwm.physics_source_defaults import PHYSICS_SELECTOR_VALUES
    for key, old in (("solar_generation", "wrf_461"), ("ysu_version", "wrf_461"),
                     ("noahmp_snow_age", "wrf_461"), ("urban_water_balance", "wrf_471")):
        default = RunConfig.__dataclass_fields__[key].default
        assert {default, old} <= set(PHYSICS_SELECTOR_VALUES[key]), key
