"""The generic input-normalization stage: the mechanism, not any one model.

These tests own the seam.  A normalization document is the only thing that
makes a source with an unreadable native grid reachable, so a document that
lies about its schema, names a method the native side does not implement,
forgets a role the stage has to bind, or plans on a field it never declares
must be refused at LOAD -- before any byte of anyone's data is touched.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwm import source_authorities, source_normalization as norm


@pytest.fixture(scope="module")
def shipped():
    """The one document this distribution ships, as its bytes."""

    path = source_authorities.packaged_normalization("icon-gdt101-pressure-v1")
    return json.loads(path.read_text(encoding="utf-8"))


def write(tmp_path, document, name="candidate.normalization.json"):
    path = tmp_path / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_the_shipped_inventory_is_closed_and_self_consistent():
    names = source_authorities.packaged_normalizer_ids()
    assert names == ("icon-d2-gdt101-model-level-v1", "icon-gdt101-pressure-v1")
    for name in names:
        spec = norm.load_normalization(name)
        assert spec.name == name
        # Every profile that declares a normalizer declares its document's pin,
        # and the pin is checked on resolve.
        profile = source_authorities.packaged_profile(spec.profile)
        assert set(profile["files"]) == set(profile["sha256"])
        assert "normalization" in profile["files"]


def test_an_unknown_normalizer_never_becomes_an_import_or_a_path():
    for name in ("some_module.run", "../../etc/passwd", "", "icon"):
        with pytest.raises(KeyError):
            source_authorities.packaged_normalization(name)
    with pytest.raises(KeyError):
        norm.normalize_packaged_inputs("some_module.run", SimpleNamespace())


def test_a_profile_must_declare_a_normalizer_name_and_its_pin_together():
    from gpuwm.source_authorities import _profile
    common = dict(source_format="grib2", mapping="a" * 64, composition="b" * 64,
                  provenance="c" * 64, data_role="x", provenance_role="y")
    with pytest.raises(ValueError, match="NAME and the"):
        _profile("stem", input_normalizer="n", **common)
    with pytest.raises(ValueError, match="NAME and the"):
        _profile("stem", normalization="d" * 64, **common)
    row = _profile("stem", input_normalizer="n", normalization="d" * 64, **common)
    assert row["files"]["normalization"] == "stem.normalization.json"
    assert row["sha256"]["normalization"] == "d" * 64
    # A profile with no normalizer keeps exactly the three published roles.
    plain = _profile("stem", **common)
    assert set(plain["files"]) == {"mapping", "composition", "provenance"}
    assert "input_normalizer" not in plain


def test_the_schema_is_checked_before_anything_is_read(tmp_path, shipped):
    document = copy.deepcopy(shipped)
    document["schema"] = "gpuwm-source-normalization-v99"
    with pytest.raises(ValueError, match="declares schema"):
        norm.load_document(write(tmp_path, document))


@pytest.mark.parametrize("key", ["name", "contract", "bridge", "source_id",
                                 "profile", "native_grid", "objects", "cadence",
                                 "roles", "plan_fields", "target", "limits",
                                 "fields", "methods"])
def test_every_required_key_is_named_when_it_is_missing(tmp_path, shipped, key):
    document = copy.deepcopy(shipped)
    document.pop(key)
    with pytest.raises(ValueError, match=f"omits the required key '{key}'"):
        norm.load_document(write(tmp_path, document))


def test_a_method_the_native_side_does_not_implement_is_refused(tmp_path, shipped):
    document = copy.deepcopy(shipped)
    document["fields"]["T"]["mode"] = "bilinear"
    with pytest.raises(ValueError, match="names remap method"):
        norm.load_document(write(tmp_path, document))


def test_an_unbindable_role_is_refused(tmp_path, shipped):
    document = copy.deepcopy(shipped)
    document["fields"]["T"]["role"] = "sea_surface"
    with pytest.raises(ValueError, match="declares role"):
        norm.load_document(write(tmp_path, document))


@pytest.mark.parametrize("role", ["geometry", "terrain", "land_fraction"])
def test_a_document_missing_a_role_the_stage_binds_is_refused(tmp_path, shipped, role):
    document = copy.deepcopy(shipped)
    for row in document["fields"].values():
        if row.get("role") == role:
            row.pop("role")
    with pytest.raises(ValueError, match=f"declares no {role} field"):
        norm.load_document(write(tmp_path, document))


def test_a_partial_selector_is_refused_rather_than_defaulted(tmp_path, shipped):
    document = copy.deepcopy(shipped)
    document["fields"]["PS"]["selector"].pop("second_level_type")
    with pytest.raises(ValueError, match="exactly the seven"):
        norm.load_document(write(tmp_path, document))


def test_an_unknown_level_ladder_kind_is_refused(tmp_path, shipped):
    document = copy.deepcopy(shipped)
    document["fields"]["T"]["levels"]["kind"] = "logarithmic"
    with pytest.raises(ValueError, match="unknown level ladder kind"):
        norm.load_document(write(tmp_path, document))


def test_planning_on_an_undeclared_field_is_refused(tmp_path, shipped):
    document = copy.deepcopy(shipped)
    document["plan_fields"] = ["CLAT", "CLON", "NOT_A_FIELD"]
    with pytest.raises(ValueError, match="plans on 'NOT_A_FIELD'"):
        norm.load_document(write(tmp_path, document))


def test_the_plan_records_must_be_declared_in_the_order_they_are_read(
        tmp_path, shipped):
    """The native side takes the three plan records positionally, and
    nothing downstream can tell a latitude record from a longitude one."""

    document = copy.deepcopy(shipped)
    document["plan_fields"] = ["CLAT", "FR_LAND", "CLON"]
    with pytest.raises(ValueError, match="order the native side reads"):
        norm.load_document(write(tmp_path, document))
    document["plan_fields"] = ["CLAT", "CLON"]
    with pytest.raises(ValueError, match="exactly three records"):
        norm.load_document(write(tmp_path, document))


def test_the_plan_command_carries_the_declared_selectors(tmp_path, shipped):
    """The converter is told which codes to look for; it compiles none in.

    This is the arbitrary acceptance test at the seam: a second source on
    the same grid template numbers its coordinate records its own way, and
    that has to be four JSON documents, never an edit to a binary.
    """

    document = copy.deepcopy(shipped)
    spec = norm.load_document(write(tmp_path, document))
    paths = {name: tmp_path / f"{name}.grib2"
             for name in document["plan_fields"]}
    tokens = norm._plan_records(spec, paths)
    assert tokens == [
        str(paths["CLAT"]), "0,191,1,1,0.0,255,0.0",
        str(paths["CLON"]), "0,191,2,1,0.0,255,0.0",
        str(paths["FR_LAND"]), "2,0,0,1,0.0,255,0.0",
    ]
    # Change the declaration and the command changes with it.  Nothing but
    # the document decides which record is the latitude.
    other = copy.deepcopy(shipped)
    other["fields"]["CLAT"]["selector"].update(
        {"category": 200, "parameter": 7})
    other["fields"]["CLON"]["selector"].update(
        {"category": 200, "parameter": 8})
    spec = norm.load_document(write(tmp_path, other, "other.json"))
    tokens = norm._plan_records(spec, paths)
    assert tokens[1] == "0,200,7,1,0.0,255,0.0"
    assert tokens[3] == "0,200,8,1,0.0,255,0.0"


def test_a_document_that_renames_itself_is_refused(tmp_path, shipped, monkeypatch):
    document = copy.deepcopy(shipped)
    document["name"] = "something-else"
    path = write(tmp_path, document)
    monkeypatch.setattr(source_authorities, "packaged_normalization",
                        lambda name: path)
    with pytest.raises(ValueError, match="calls itself"):
        norm.load_normalization("icon-gdt101-pressure-v1")


@pytest.mark.parametrize("ladder,level,expected", [
    ({"kind": "scaled", "scale": 100.0, "values": [850]}, 850,
     (0, 0, 0, 100, 85000.0, 255, 0.0)),
    ({"kind": "table", "values": {"5": 0.005}}, 5,
     (0, 0, 0, 100, 0.005, 255, 0.0)),
    ({"kind": "bounds", "values": {"3": [0.03, 0.09]}}, 3,
     (0, 0, 0, 100, 0.03, 255, 0.09)),
])
def test_each_level_ladder_kind_resolves_its_own_selector(
        tmp_path, shipped, ladder, level, expected):
    document = copy.deepcopy(shipped)
    row = document["fields"]["T"]
    row["levels"] = ladder
    spec = norm.load_document(write(tmp_path, document))
    assert spec.selector_for("T", level) == expected


def test_a_level_ladder_with_no_level_refuses_instead_of_guessing(tmp_path, shipped):
    spec = norm.load_document(write(tmp_path, copy.deepcopy(shipped)))
    for level in (None, 775):
        with pytest.raises(ValueError, match="declared ladder"):
            spec.selector_for("T", level)


def test_the_stage_reads_its_limits_from_the_document(tmp_path, shipped):
    document = copy.deepcopy(shipped)
    document["limits"]["max_input_files"] = 2
    spec = norm.load_document(write(tmp_path, document))
    names = [
        "icon_global_icosahedral_time-invariant_2026091500_CLAT.grib2",
        "icon_global_icosahedral_time-invariant_2026091500_CLON.grib2",
        "icon_global_icosahedral_time-invariant_2026091500_HSURF.grib2",
    ]
    with pytest.raises(ValueError, match="bounded file count"):
        norm.validate_inventory(spec, [tmp_path / n for n in names])


def test_the_target_envelope_comes_from_the_document(tmp_path, shipped):
    document = copy.deepcopy(shipped)
    document["target"]["step_degrees"] = 0.25
    document["target"]["max_abs_latitude"] = 60.0
    spec = norm.load_document(write(tmp_path, document))
    window = norm.target_from_points(spec, [40.0, 41.0], [-99.0, -98.0])
    assert window.dx == 0.25 and window.dy == 0.25
    with pytest.raises(ValueError, match="beyond 60"):
        norm.target_from_points(spec, [70.0, 71.0], [-99.0, -98.0])
    # And the declared spacing is the only spacing the window accepts.
    with pytest.raises(ValueError, match="spacing is fixed at 0.25"):
        norm.TargetWindow(spec, 0.0, 0.0, 2, 2, 0.125, 0.125)


def test_a_cycle_hour_outside_the_declared_grid_is_named(tmp_path, shipped):
    spec = norm.load_document(write(tmp_path, copy.deepcopy(shipped)))
    with pytest.raises(ValueError, match="cycles are 00, 06, 12, 18 UTC"):
        norm.parse_object(
            spec, tmp_path /
            "icon_global_icosahedral_time-invariant_2026091503_CLAT.grib2")
    with pytest.raises(ValueError, match="not 03"):
        spec.horizon_hours(3)
