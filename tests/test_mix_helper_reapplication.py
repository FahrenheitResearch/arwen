"""Additional prepared stock-key/default reconciliation cases.

Install beside test_mix_selection_byte_controls.py after the forward merge.
These cases have been inspected by AST only, not executed on a candidate.
"""
import tomllib

import pytest

from gpuwm.physics_source_defaults import (
    PHYSICS_SELECTOR_VALUES, namelist_physics_defaults, recipe_physics_defaults)
from test_mix_selection_byte_controls import (
    build_experiment, documents, import_documents, named_documents, set_column)


@pytest.mark.parametrize("parsed", (False, True))
@pytest.mark.parametrize("named,values,expected", (
    (True, ".true., .true.", [True, True]),
    (True, ".false., .false.", [False, False]),
    (True, ".true.", [True, False]),
    (True, ".false.", [False, False]),
    (False, ".true.", [True, True]),
    (False, ".false.", [True, True]),
))
def test_emitted_stock_mix_binding_survives_source_default_helper(
        named, values, expected, parsed, tmp_path):
    row = named_documents("explicit_true") if named else documents("explicit_true")
    row = set_column(row, "mix_full_fields", values)
    text, report = import_documents(tmp_path, row, named=named, parsed=parsed)
    raw = tomllib.loads(text)
    exp = build_experiment(raw, source="stock mixing binding")
    assert [domain.run.mix_full_fields for domain in exp.domains] == expected
    if named and expected[0]:
        assert raw["shared"].get("mix_full_fields", True) is True
    if not named and values == ".false.":
        assert "mix_full_fields" not in raw["shared"]
        assert all("mix_full_fields" not in domain for domain in raw["domain"])
    for substitution in report.substitutions:
        if substitution.key == "mix_full_fields":
            assert "no perturbation-field mixing branch" not in substitution.reason


def test_stock_mix_default_is_table_scoped_and_not_a_generation_comment():
    assert "mix_full_fields" not in PHYSICS_SELECTOR_VALUES
    assert namelist_physics_defaults("hrrr_wrf.nl")["mix_full_fields"] is False
    assert "mix_full_fields" not in namelist_physics_defaults("namelist.input")
    for source in ("hrrr", "hrrr-native", "hrrr-prs"):
        assert recipe_physics_defaults(source)["mix_full_fields"] is False
    for source in ("gfs", "era5", "rap", "rrfs"):
        assert "mix_full_fields" not in recipe_physics_defaults(source)


@pytest.mark.parametrize("parsed", (False, True))
@pytest.mark.parametrize("request_source,values,expected", (
    ("hrrr", ".false., .false.", [False, False]),
    ("hrrr", ".true., .true.", [True, True]),
    ("hrrr", ".true.", [True, False]),
    ("gfs", ".false., .false.", [True, True]),
    (None, ".false., .false.", [True, True]),
))
def test_authored_source_request_survives_arbitrary_output_names(
        request_source, values, expected, parsed, tmp_path):
    from gpuwm.namelist_import import import_namelists, import_parsed_namelists
    from gpuwm.fortran_namelist import parse_namelist_text
    row = set_column(named_documents("explicit_true"), "mix_full_fields", values)
    wps = tmp_path / "authored.wps"
    inp = tmp_path / "authored.input"
    options = {"request_source": request_source}
    if parsed:
        text, _ = import_parsed_namelists(
            parse_namelist_text(row["wps"]), parse_namelist_text(row["input"]),
            wps_path=wps, input_path=inp, **options)
    else:
        wps.write_text(row["wps"], encoding="utf-8")
        inp.write_text(row["input"], encoding="utf-8")
        text, _ = import_namelists(wps, inp, **options)
    exp = build_experiment(tomllib.loads(text), source="authored request binding")
    assert [domain.run.mix_full_fields for domain in exp.domains] == expected
