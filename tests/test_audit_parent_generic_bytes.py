"""Candidate byte controls generated only by original audit-parent emitters.

Copy this test into tests/ and the matrix plus generated historical output
folders into tests/fixtures/generic_byte_controls/. Generation receipts must
name the old Git objects; candidate runs only consume those existing pins.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tomllib

import pytest

from gpuwm.domain_wizard import render_config
from gpuwm.namelist_import import import_namelists


FIXTURES = Path(__file__).parent / "fixtures/generic_byte_controls"
PARENTS = (
    "425e3e068012eafae8d9e6f4ccfb1249151ae91e",  # sfclay audit parent
    "1077c6e8abe502aab59de103d7e6d70dbb8588a1",  # fixed and vadv audit parent
    "32261e42e85fadb9fa4b934f65aba0a22586d5f1",  # dycore audit parent
)
NAMELISTS = (
    "omitted_legacy_defaults", "explicit_vertical_order3",
    "ordinary_fixed60_triplet", "ruc_monthly_omitted", "ruc_monthly_false0",
)
RECIPES = ("gfs", "era5", "rap", "rrfs")


def controls(parent):
    payload = (FIXTURES / "generic-matrix.json").read_bytes()
    pins = json.loads((FIXTURES / parent / "historical-pins.json").read_bytes())
    assert pins["mode"] == "baseline"
    assert pins["audit_parent"] == parent
    assert hashlib.sha256(payload).hexdigest() == pins["matrix_sha256"]
    assert set(pins["cases"]) == {
        *("namelist/" + name for name in NAMELISTS),
        *("recipe/" + source for source in RECIPES),
    }
    return json.loads(payload), pins


def assert_old_bytes(parent, pins, key, candidate):
    expected = pins["cases"][key]
    path = Path(expected["file"])
    assert len(path.parts) == 1 and path.suffix == ".toml"
    old = (FIXTURES / parent / path).read_bytes()
    assert len(old) == expected["bytes"]
    assert hashlib.sha256(old).hexdigest() == expected["sha256"]
    assert candidate.encode("utf-8") == old


#: Lines a recipe emission gained after these parents were pinned, by
#: source.  e19441c1e gave RAP the operational fork's request row (RAP runs
#: the same WRF 3.9 fork as HRRR), so an authored rap recipe states the
#: fork's surface-layer generation in [shared].  The matrix recipe is not
#: mp_physics = 28, so the fork Thompson keys stay scoped out of it.
FORK_GENERATION_LINES = {"rap": ('mynn_sfclay_variant = "gsl_wrf39"\n',)}


def assert_old_recipe_bytes(parent, pins, source, candidate):
    """The parent's recipe bytes, plus exactly the fork-generation lines."""
    lines = candidate.splitlines(keepends=True)
    shared = tomllib.loads(candidate)["shared"]
    for line in FORK_GENERATION_LINES.get(source, ()):
        assert lines.count(line) == 1, line
        key = line.split(" = ")[0]
        assert shared[key] == tomllib.loads(line)[key], line
        lines.remove(line)
    assert_old_bytes(parent, pins, "recipe/" + source, "".join(lines))


@pytest.mark.parametrize("parent", PARENTS)
@pytest.mark.parametrize("case", NAMELISTS)
def test_non_hrrr_imported_config_bytes_match_original_audit_parent(parent, case, tmp_path):
    matrix, pins = controls(parent)
    documents = matrix["namelists"][case]
    wps, inp = tmp_path / "namelist.wps", tmp_path / "namelist.input"
    wps.write_text(documents["wps"], encoding="utf-8")
    inp.write_text(documents["input"], encoding="utf-8")
    emitted, _ = import_namelists(wps, inp)
    assert_old_bytes(parent, pins, "namelist/" + case, emitted)


@pytest.mark.parametrize("parent", PARENTS)
@pytest.mark.parametrize("source", RECIPES)
def test_non_hrrr_recipe_bytes_match_original_audit_parent(parent, source):
    matrix, pins = controls(parent)
    recipe = matrix["recipes"]
    emitted = render_config(
        name=recipe["name"], start_time=datetime.fromisoformat(recipe["start_time"]),
        hours=recipe["hours"], projection=recipe["projection"],
        dims=[tuple(row) for row in recipe["dims"]], ratios=tuple(recipe["ratios"]),
        root_dx_m=recipe["root_dx_m"], fetch_hints={"source": source},
        case_data=None, profile=recipe["profile"],
    )
    assert_old_recipe_bytes(parent, pins, source, emitted)
