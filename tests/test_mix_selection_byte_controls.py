"""Draft regressions for explicit mixing choices and historical omission bytes.

Prepared by AST inspection only. Historical controls are generated separately
by the exact old parent, never by the candidate executing these tests.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tomllib

import pytest

from gpuwm.companion_domains import candidate_wps_text
from gpuwm.experiment import build_experiment_from_config_tables
from gpuwm.fortran_namelist import parse_namelist_text
from gpuwm.hrrr_route_inputs import render_namelist_input
from gpuwm.namelist_import import import_namelists, import_parsed_namelists
from gpuwm.physics_source_defaults import (
    read_physics_selector_comment, with_physics_selector_comment)


PARENT = "1077c6e8abe502aab59de103d7e6d70dbb8588a1"
FIXTURES = Path(__file__).parent / "fixtures/mix_parent_byte_controls"
CASES = ("explicit_true", "omitted", "short_true")


def build_experiment(raw, *, source):
    return build_experiment_from_config_tables(raw, source=source, base_dir=Path.cwd())


def matrix():
    payload = (FIXTURES / "matrix.json").read_bytes()
    document = json.loads(payload)
    assert document["schema"] == "mix-parent-byte-inputs-v1"
    assert document["audit_parent"] == PARENT
    assert set(document["cases"]) == set(CASES)
    return document, hashlib.sha256(payload).hexdigest()


def documents(case):
    document, _ = matrix()
    return deepcopy(document["cases"][case])


def import_documents(tmp_path, row, *, parsed=False, named=False, comment=False):
    wps_path = tmp_path / "namelist.wps"
    input_path = tmp_path / ("hrrr_wrf.nl" if named else "namelist.input")
    text = row["input"]
    if comment:
        text = with_physics_selector_comment(text, {"terrain_clock": "measured"})
    if parsed:
        assert not wps_path.exists() and not input_path.exists()
        return import_parsed_namelists(
            parse_namelist_text(row["wps"]), parse_namelist_text(text),
            wps_path=wps_path, input_path=input_path)
    wps_path.write_text(row["wps"], encoding="utf-8")
    input_path.write_text(text, encoding="utf-8")
    return import_namelists(wps_path, input_path)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("parsed", (False, True))
def test_generic_mix_bytes_match_exact_original_parent(case, parsed, tmp_path):
    _, matrix_sha = matrix()
    pins = json.loads((FIXTURES / "historical-pins.json").read_bytes())
    assert pins["mode"] == "baseline"
    assert pins["audit_parent"] == PARENT
    assert pins["matrix_sha256"] == matrix_sha
    expected = pins["cases"][case]
    relative = Path(expected["file"])
    assert len(relative.parts) == 1 and relative.suffix == ".toml"
    old = (FIXTURES / relative).read_bytes()
    assert len(old) == expected["bytes"]
    assert hashlib.sha256(old).hexdigest() == expected["sha256"]
    emitted, report = import_documents(tmp_path, documents(case), parsed=parsed)
    assert emitted.encode("utf-8") == old
    exp = build_experiment(tomllib.loads(emitted), source="generic mixing bytes")
    assert [domain.run.mix_full_fields for domain in exp.domains] == [True, True]
    substitutions = [row for row in report.substitutions if row.key == "mix_full_fields"]
    assert bool(substitutions) is (case != "explicit_true")


def set_column(row, key, value):
    lines = row["input"].splitlines(keepends=True)
    index = next(i for i, line in enumerate(lines) if line.lstrip().startswith(key + " ="))
    lines[index] = " " + key + " = " + value + ",\n"
    row["input"] = "".join(lines)
    return row


def named_documents(case="omitted"):
    row = documents(case)
    # A valid synthetic MYNN pair permits the named generation selected by
    # the existing request table. This test owns only its mixing choice.
    row = set_column(row, "mp_physics", "8, 8")
    row = set_column(row, "sf_sfclay_physics", "5, 5")
    return set_column(row, "bl_pbl_physics", "5, 5")


@pytest.mark.parametrize("parsed", (False, True))
@pytest.mark.parametrize("case,expected", (
    ("omitted", [False, False]),
    ("short_true", [True, False]),
    ("explicit_true", [True, True]),
))
def test_named_mix_defaults_apply_only_to_unsupplied_elements(case, expected, parsed, tmp_path):
    text, report = import_documents(tmp_path, named_documents(case), parsed=parsed, named=True)
    exp = build_experiment(tomllib.loads(text), source="named mixing selection")
    assert [domain.run.mix_full_fields for domain in exp.domains] == expected
    assert not [row for row in report.substitutions if row.key == "mix_full_fields"]


@pytest.mark.parametrize("named", (False, True))
@pytest.mark.parametrize("values,expected", (
    (".false., .true.", [False, True]),
    (".true., .false.", [True, False]),
    (".false., .false.", [False, False]),
    (".true., .true.", [True, True]),
))
def test_named_mix_and_generic_resolution_survive_the_typed_selector_comment(named, values, expected, tmp_path):
    row = named_documents("explicit_true") if named else documents("explicit_true")
    set_column(row, "mix_full_fields", values)
    text, report = import_documents(tmp_path, row, named=named, comment=True)
    exp = build_experiment(tomllib.loads(text), source="explicit mixing column")
    assert [domain.run.mix_full_fields for domain in exp.domains] == (
        expected if named else [True, True])
    assert exp.root.run.terrain_clock == "measured"
    substitutions = [row for row in report.substitutions if row.key == "mix_full_fields"]
    assert bool(substitutions) is (not named and not all(expected))


@pytest.mark.parametrize("diffusion,expected", (
    ("1, 1", [False, False]),
    ("2, 1", [True, False]),
    ("1, 2", [False, True]),
))
def test_generic_unsupplied_mix_tail_preserves_each_diffusion_operator(diffusion, expected, tmp_path):
    row = set_column(documents("omitted"), "diff_opt", diffusion)
    text, _ = import_documents(tmp_path, row, parsed=True)
    exp = build_experiment(tomllib.loads(text), source="per-domain diffusion defaults")
    assert [domain.run.mix_full_fields for domain in exp.domains] == expected


@pytest.mark.parametrize("values,expected", (
    (".false., .true.", [False, True]),
    (".true., .false.", [True, False]),
))
def test_actual_generated_namelist_reimport_preserves_mix_choices(values, expected, tmp_path):
    row = set_column(named_documents("explicit_true"), "mix_full_fields", values)
    text, _ = import_documents(tmp_path, row, named=True, comment=True)
    raw = tomllib.loads(text)
    exp = build_experiment(raw, source="generated mixing selection")
    wps = tmp_path / "emitted.wps"
    inp = tmp_path / "hrrr_wrf.nl.generated"
    wps.write_text(candidate_wps_text(raw, exp, exp, tmp_path / "input.toml"), encoding="utf-8")
    generated = render_namelist_input(exp)
    carried = read_physics_selector_comment(generated)
    assert carried["mynn_sfclay_variant"] == exp.root.run.mynn_sfclay_variant
    inp.write_text(generated, encoding="utf-8")
    restored_text, _ = import_namelists(wps, inp,
        rrtmg_variant=exp.root.run.ra_rrtmg_variant,
        rrtmg_compatibility=exp.root.run.wrf_rrtmg_compatibility)
    restored = build_experiment(tomllib.loads(restored_text), source="actual mixing reimport")
    assert [domain.run.mix_full_fields for domain in restored.domains] == expected
