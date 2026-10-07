"""Albsol off bytes retain the old parent and the approved prior zero fix."""
from datetime import datetime
import hashlib
import tomllib

import pytest

from gpuwm.domain_wizard import render_config
from gpuwm.experiment import build_experiment
from gpuwm.namelist_import import import_namelists
from test_audit_parent_generic_bytes import (
    FIXTURES, assert_old_bytes, assert_old_recipe_bytes, controls)


ALBSOL_PARENT = "055595b3a8e019be508f05991b67af607dcee338"
ORIGINAL_AUDIT_PARENT = "1077c6e8abe502aab59de103d7e6d70dbb8588a1"
CASES = (
    "namelist/omitted_legacy_defaults", "namelist/explicit_vertical_order3",
    "namelist/ordinary_fixed60_triplet", "namelist/ruc_monthly_omitted",
    "namelist/ruc_monthly_false0", "recipe/gfs", "recipe/era5", "recipe/rap", "recipe/rrfs",
)


def emit_namelist(tmp_path, documents, extra=""):
    wps, inp = tmp_path / "namelist.wps", tmp_path / "namelist.input"
    wps.write_text(documents["wps"], encoding="utf-8")
    inp.write_text(documents["input"].replace("&physics\n", "&physics\n" + extra),
                   encoding="utf-8")
    emitted, _ = import_namelists(wps, inp)
    return emitted


def assert_old_parent_zero_line_was_the_exact_inherited_difference(key, pins):
    _, original_pins = controls(ORIGINAL_AUDIT_PARENT)
    record = pins["cases"][key]
    old = (FIXTURES / ALBSOL_PARENT / record["file"]).read_bytes()
    original_record = original_pins["cases"][key]
    original = (FIXTURES / ORIGINAL_AUDIT_PARENT / original_record["file"]).read_bytes()
    assert hashlib.sha256(old).hexdigest() == record["sha256"]
    assert len(old) == record["bytes"]
    assert hashlib.sha256(original).hexdigest() == original_record["sha256"]
    assert len(original) == original_record["bytes"]
    zero_line = b"fractional_seaice = 0\n"
    assert old.count(zero_line) == 1
    assert len(old) - len(original) == 22
    assert old.replace(zero_line, b"") == original
    return original_pins


@pytest.mark.parametrize("key", CASES)
def test_albsol_generic_bytes_retain_parent_with_prior_fractional_zero_fix(key, tmp_path):
    matrix, pins = controls(ALBSOL_PARENT)
    if key.startswith("namelist/"):
        emitted = emit_namelist(tmp_path, matrix["namelists"][key.split("/", 1)[1]])
    else:
        recipe = matrix["recipes"]
        source = key.split("/", 1)[1]
        emitted = render_config(
            name=recipe["name"], start_time=datetime.fromisoformat(recipe["start_time"]),
            hours=recipe["hours"], projection=recipe["projection"],
            dims=[tuple(row) for row in recipe["dims"]], ratios=tuple(recipe["ratios"]),
            root_dx_m=recipe["root_dx_m"], fetch_hints={"source": source},
            case_data=None, profile=recipe["profile"],
        )
    assert "alb_sol" not in tomllib.loads(emitted)["shared"]
    if key == "namelist/ruc_monthly_false0":
        # Parent055 already inherited a vegetation zero-line regression.
        # Vadv intentionally corrected it to genuine pre-lane audit1077 bytes.
        original_pins = assert_old_parent_zero_line_was_the_exact_inherited_difference(key, pins)
        assert_old_bytes(ORIGINAL_AUDIT_PARENT, original_pins, key, emitted)
    elif key.startswith("recipe/"):
        assert_old_recipe_bytes(ALBSOL_PARENT, pins, key.split("/", 1)[1], emitted)
    else:
        assert_old_bytes(ALBSOL_PARENT, pins, key, emitted)


@pytest.mark.parametrize("option", ("alb_sol", "aer_opt", "swint_opt"))
def test_explicit_zero_radiation_driver_options_keep_old_generic_bytes(option, tmp_path):
    matrix, pins = controls(ALBSOL_PARENT)
    documents = matrix["namelists"]["omitted_legacy_defaults"]
    omitted = emit_namelist(tmp_path, documents)
    explicit = emit_namelist(tmp_path, documents, f" {option} = 0,\n")
    assert explicit == omitted
    assert option not in tomllib.loads(explicit)["shared"]
    assert_old_bytes(ALBSOL_PARENT, pins, "namelist/omitted_legacy_defaults", explicit)
    experiment = build_experiment(tomllib.loads(explicit), source="generic radiation zero control")
    for domain in experiment.domains:
        assert domain.run.alb_sol == domain.run.aer_opt == domain.run.swint_opt == 0
