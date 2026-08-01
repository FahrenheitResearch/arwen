"""``gpuwm dual-run``: one row per capsule field, and no field exempt.

The 5090 carries no ECC, so a bit that flips in VRAM is reported by nothing.
Running the same configuration twice and comparing the two capsules is the
detector.  A detector with an ignore list is a detector with a blind spot, so
this suite mutates *every* leaf of a matched capsule -- every pin, every
schedule entry, every inventory entry, every scalar, every wrfout SHA-256,
every per-domain trajectory digest -- and requires the shipped command to
refuse each one by name.

The table is not a list anybody maintains: it is generated from the capsule
itself, and a meta-test holds that generated set against the capsule schema,
so a field added to the schema later cannot escape it.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import certification_fixtures as fixtures
from gpuwm.certify.band import write_certification_json
from gpuwm.certify.capsule import SCHEMA_PATH
from gpuwm.certify.dualrun import (capsule_field_paths, compare_capsule_files,
                                   compare_capsules, delete_leaf, leaf_value,
                                   set_leaf, split_path)
from gpuwm.cli import main

CAPSULE = fixtures.matched_capsule(fixtures.shipped_band()["config_sha256"])
FIELD_PATHS = capsule_field_paths(CAPSULE)


def _mutate(value):
    """A type-preserving edit, so exactly one leaf path moves."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, str):
        return value + "-mutated"
    if isinstance(value, (int, float)):
        return value + 1
    if value is None:
        return 0
    if isinstance(value, dict):
        return {"mutated": True}
    if isinstance(value, list):
        return [0]
    raise AssertionError(f"no mutation defined for {type(value)!r}")


def _write(path: Path, document) -> Path:
    return write_certification_json(path, document)


def test_two_matched_capsules_are_identical(tmp_path, capsys):
    a = _write(tmp_path / "a.json", CAPSULE)
    b = _write(tmp_path / "b.json", copy.deepcopy(CAPSULE))
    assert main(["dual-run", "--capsule-a", str(a),
                 "--capsule-b", str(b)]) == 0
    assert "identical" in capsys.readouterr().out


def test_the_mutation_table_is_not_empty_and_covers_the_named_field_kinds():
    """A vacuous table would make every row below pass for free."""
    assert len(FIELD_PATHS) > 40, FIELD_PATHS
    kinds = {
        "a pin": "numerical_stack.numpy_version.status",
        "a schedule entry": "run_shape.run_seconds",
        "an inventory entry": "input_bytes.entries.geography_root.sha256",
        "a scalar": "code.worktree_clean",
        "a wrfout SHA-256": "output.frames[0].sha256",
        "a trajectory digest": "output.trajectory_digest.d01",
    }
    missing = {kind: path for kind, path in kinds.items()
               if path not in FIELD_PATHS}
    assert not missing, missing


@pytest.mark.parametrize("field_path", FIELD_PATHS)
def test_every_single_field_mutation_is_refused_by_name(tmp_path, field_path):
    mutated = copy.deepcopy(CAPSULE)
    set_leaf(mutated, field_path, _mutate(leaf_value(CAPSULE, field_path)))

    a = _write(tmp_path / "a.json", CAPSULE)
    b = _write(tmp_path / "b.json", mutated)

    comparison = compare_capsule_files(a, b)
    assert comparison.identical is False
    assert comparison.first_divergent_field == field_path
    assert len(comparison.divergences) == 1
    assert main(["dual-run", "--capsule-a", str(a),
                 "--capsule-b", str(b)]) != 0


@pytest.mark.parametrize("field_path", FIELD_PATHS)
def test_deleting_a_field_is_refused_naming_it_or_the_hole_it_left(field_path):
    """A dropped field and a field carrying null are different claims.

    Removing the last leaf of a section leaves an empty container, which is
    itself a leaf and sorts before its own former child -- so the reported
    first divergent field is that container.  Either way the deleted path is
    among the divergences and the comparison refuses.
    """
    if isinstance(split_path(field_path)[-1], int):
        pytest.skip("a list element is dropped by reindexing, not by key")
    mutated = copy.deepcopy(CAPSULE)
    delete_leaf(mutated, field_path)
    comparison = compare_capsules(CAPSULE, mutated)
    assert comparison.identical is False
    reported = comparison.first_divergent_field
    assert reported == field_path or field_path.startswith(reported + ".")
    assert field_path in {item.field_path for item in comparison.divergences}


def test_the_field_set_is_generated_from_the_capsule_schema():
    """A field added to the schema later cannot escape the table."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    top_level = set(schema["required"])
    covered = {path.split(".")[0].split("[")[0] for path in FIELD_PATHS}
    assert top_level <= covered, sorted(top_level - covered)

    pins = set(schema["properties"]["numerical_stack"]["required"])
    pin_paths = {path.split(".")[1] for path in FIELD_PATHS
                 if path.startswith("numerical_stack.")}
    assert pins <= pin_paths, sorted(pins - pin_paths)

    frames = {path for path in FIELD_PATHS
              if path.startswith("output.frames[")}
    assert len([path for path in frames if path.endswith(".sha256")]) >= 2
    digests = {path for path in FIELD_PATHS
               if path.startswith("output.trajectory_digest.")}
    assert len(digests) >= 2


def test_an_absent_field_is_not_a_null_field():
    absent = copy.deepcopy(CAPSULE)
    del absent["registry"]["template_id"]
    nulled = copy.deepcopy(CAPSULE)
    nulled["registry"]["template_id"] = None
    assert compare_capsules(absent, nulled).identical is False
    assert compare_capsules(absent, nulled).first_divergent_field == (
        "registry.template_id")


def test_an_emptied_container_does_not_vanish_from_the_comparison():
    emptied = copy.deepcopy(CAPSULE)
    emptied["receipts"] = {}
    comparison = compare_capsules(CAPSULE, emptied)
    assert comparison.identical is False
    assert comparison.first_divergent_field.startswith("receipts")


def test_the_comparison_order_is_a_property_of_the_documents(tmp_path):
    """Two mutations, and the reported first field does not depend on order."""
    mutated = copy.deepcopy(CAPSULE)
    set_leaf(mutated, "code.git_commit", "z" * 40)
    set_leaf(mutated, "output.frames[1].sha256", "z" * 64)
    forward = compare_capsules(CAPSULE, mutated).first_divergent_field
    shuffled = json.loads(json.dumps(mutated, sort_keys=False))
    reordered = {key: shuffled[key] for key in reversed(list(shuffled))}
    assert compare_capsules(CAPSULE, reordered).first_divergent_field == (
        forward)
    assert forward == "code.git_commit"
