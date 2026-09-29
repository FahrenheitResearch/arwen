"""The FTZ claim census covers the tree, and it can be made to fail.

The mutations below drive the ways a census goes stale: a claim sentence
nobody registered, a registered one removed, reworded, copied or moved to
another file, a token that contradicts the receipt, a hash edited by hand,
and a stated total that disagrees with the records.  A census gate that
survives any of them is decoration.

Lines added or removed above a claim must NOT fail it.  A row added to
CHANGELOG.md moves every claim below it and changes none of them; while the
line number was part of a site's key that failed the gate, so every branch
that added a row rewrote the census and two such branches conflicted in it.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.ftz_receipt import claim_census as cc  # noqa: E402
from tools.ftz_receipt import probe  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CENSUS = ROOT / "gpuwm" / "verify" / "ftz_claim_sites.json"
RECEIPT = ROOT / "tools" / "ftz_receipt" / "receipt" / "receipt.json"


@pytest.fixture(scope="module")
def census() -> dict:
    return json.loads(CENSUS.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def receipt() -> dict:
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def test_census_covers_the_public_tree(census, receipt):
    assert census["schema"] == cc.SCHEMA_ID
    assert cc.check_census(ROOT, census, receipt) == []


def test_scope_excludes_vendor_and_release_excluded_paths(census):
    globs = cc.release_exclusions(ROOT)
    assert globs, "RELEASE-EXCLUDE.txt did not parse"
    for record in census["sites"]:
        assert not cc.is_excluded(record["file"], globs), record["file"]
    assert cc.is_excluded("tools/rustwx/vendor/x.rs", globs)
    assert cc.is_excluded("handoffs/anything.md", globs)
    # The mapped-engine port vendored a fourth tree.  Its crates-io mirror
    # and donor snapshot are read-only, so a record pinned inside one would
    # fail on the next re-vendor with no fix available short of editing
    # somebody else's source.  The workspace's OWN crates stay in scope.
    assert cc.is_excluded("tools/rw_wps/vendor/crates-io/x/src/lib.rs", globs)
    assert not cc.is_excluded(
        "tools/rw_wps/crates/mapped-engine/src/main.rs", globs)


def test_tests_are_in_scope(census):
    assert any(record["file"].startswith("tests/")
               for record in census["sites"]), (
        "the census must cover tests/; a claim in a test is still a claim")


def test_behavioural_sites_are_registered_not_edited(census):
    registered = {record["file"] for record in census["sites"]
                  if record["kind"] == "behavioural"}
    for path in cc.BEHAVIOURAL_FILES:
        assert path in registered, path
    diff = subprocess.run(
        ["git", "diff", "HEAD", "--stat", "--"] + list(cc.BEHAVIOURAL_FILES),
        cwd=str(ROOT), capture_output=True, text=True, check=True)
    assert diff.stdout.strip() == "", (
        "a behavioural site changed; revisiting one is [D-36], not this "
        "package:\n" + diff.stdout)


def test_unattributed_records_carry_no_token(census):
    for record in census["sites"]:
        if record["attribution"] == cc.ATTRIBUTION_PENDING:
            assert record["asserted_token"] is None, record


# ---- mutation controls ----------------------------------------------------

def _tiny_tree(tmp_path: Path) -> Path:
    root = tmp_path / "tree"
    (root / "gpuwm" / "verify").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "RELEASE-EXCLUDE.txt").write_text("handoffs/**\n", encoding="utf-8")
    (root / "docs" / "claims.md").write_text(
        "The loader supplies no FTZ option.\n"
        "Nothing here mentions the other thing.\n"
        "A subnormal survives the multiply.\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=str(root), check=True,
                   capture_output=True)
    subprocess.run(["git", "add", "-A"], cwd=str(root), check=True,
                   capture_output=True)
    return root


@pytest.fixture()
def tiny(tmp_path: Path):
    root = _tiny_tree(tmp_path)
    document = cc.build_census(root)
    assert cc.check_census(root, document) == []
    return root, document


def _stub_receipt(route: str, mechanism: str, verdict: str) -> dict:
    return {"cells": [{"route": route, "mechanism": mechanism,
                       "verdict": verdict}]}


def test_control_unregistered_claim_sentence_fails(tiny):
    root, document = tiny
    path = root / "docs" / "claims.md"
    path.write_text(path.read_text(encoding="utf-8")
                    + "An unregistered sentence about FTZ.\n",
                    encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=str(root), check=True,
                   capture_output=True)
    problems = cc.check_census(root, document)
    assert any("unregistered claim" in problem for problem in problems), \
        problems


def test_control_edited_anchor_text_fails(tiny):
    root, document = tiny
    path = root / "docs" / "claims.md"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] = "The loader supplies an FTZ option after all."
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=str(root), check=True,
                   capture_output=True)
    problems = cc.check_census(root, document)
    assert any("anchor text changed" in problem for problem in problems), \
        problems


def test_control_flipped_token_fails(tiny):
    root, document = tiny
    verdicts = [item for item in probe.VERDICTS
                if item != probe.VERDICT_NOT_APPLICABLE]
    record = document["sites"][0]
    record["route"] = "R1"
    record["mechanism"] = probe.MECHANISMS[0][0]
    record["asserted_token"] = verdicts[0]
    record["attribution"] = cc.ATTRIBUTION_SET
    agreeing = _stub_receipt("R1", probe.MECHANISMS[0][0], verdicts[0])
    assert cc.check_census(root, document, agreeing) == []

    record["asserted_token"] = verdicts[1]
    problems = cc.check_census(root, document, agreeing)
    assert any("but the receipt's cell says" in problem
               for problem in problems), problems


def test_control_token_for_a_cell_the_receipt_lacks_fails(tiny):
    root, document = tiny
    record = document["sites"][0]
    record["route"] = "R9"
    record["mechanism"] = "not a mechanism"
    record["asserted_token"] = probe.VERDICT_IEEE
    record["attribution"] = cc.ATTRIBUTION_SET
    problems = cc.check_census(
        root, document, _stub_receipt("R1", probe.MECHANISMS[0][0],
                                      probe.VERDICT_IEEE))
    assert any("the receipt does not carry" in problem
               for problem in problems), problems


# ---- line movement: the census follows the text, not the line -------------

ROW_ABOVE = ("## 9.9.9\n"
             "\n"
             "- A new release row that makes no claim.\n"
             "\n")


def _git_add(root: Path) -> None:
    subprocess.run(["git", "add", "-A"], cwd=str(root), check=True,
                   capture_output=True)


def _claims(root: Path) -> Path:
    return root / "docs" / "claims.md"


def test_control_row_added_above_the_claims_passes(tiny):
    # A release-note row landing above registered claims moves each of them
    # down and changes none.  Keyed on line numbers this failed the gate, so
    # every branch that added a CHANGELOG row had to regenerate the census
    # and every merge of two such branches conflicted in it.
    root, document = tiny
    path = _claims(root)
    path.write_text(ROW_ABOVE + path.read_text(encoding="utf-8"),
                    encoding="utf-8")
    _git_add(root)
    assert cc.check_census(root, document) == []


def test_control_line_removed_above_a_claim_passes(tiny):
    root, document = tiny
    path = _claims(root)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[1] == "Nothing here mentions the other thing."
    del lines[1]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _git_add(root)
    assert cc.check_census(root, document) == []


def test_moved_claim_keeps_its_token_and_regeneration_moves_its_hint(tiny):
    root, document = tiny
    verdicts = [item for item in probe.VERDICTS
                if item != probe.VERDICT_NOT_APPLICABLE]
    record = document["sites"][1]
    record["route"] = "R1"
    record["mechanism"] = probe.MECHANISMS[0][0]
    record["asserted_token"] = verdicts[0]
    record["attribution"] = cc.ATTRIBUTION_SET
    agreeing = _stub_receipt("R1", probe.MECHANISMS[0][0], verdicts[0])
    path = _claims(root)
    path.write_text(ROW_ABOVE + path.read_text(encoding="utf-8"),
                    encoding="utf-8")
    _git_add(root)

    notes: list[str] = []
    assert cc.check_census(root, document, agreeing, notes=notes) == []
    assert notes and notes[0].startswith("2 of 2 sites"), notes

    rebuilt = cc.build_census(root, document)
    assert cc.check_census(root, rebuilt, agreeing) == []
    moved = [item for item in rebuilt["sites"]
             if item["anchor"] == record["anchor"]]
    assert [item["line"] for item in moved] == [record["line"] + 4]
    assert moved[0]["asserted_token"] == verdicts[0]

    record["asserted_token"] = verdicts[1]
    problems = cc.check_census(root, document, agreeing)
    assert any("but the receipt's cell says" in problem
               for problem in problems), problems


def test_control_removed_claim_fails(tiny):
    root, document = tiny
    path = _claims(root)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[2] == "A subnormal survives the multiply."
    del lines[2]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _git_add(root)
    problems = cc.check_census(root, document)
    assert any("registered claim no longer present" in problem
               for problem in problems), problems


def test_control_reworded_claim_below_a_new_row_fails(tiny):
    root, document = tiny
    path = _claims(root)
    text = path.read_text(encoding="utf-8")
    assert "A subnormal survives the multiply." in text
    text = text.replace("A subnormal survives the multiply.",
                        "A subnormal is flushed by the multiply.")
    path.write_text(ROW_ABOVE + text, encoding="utf-8")
    _git_add(root)
    problems = cc.check_census(root, document)
    assert any("A subnormal is flushed by the multiply." in problem
               for problem in problems), problems


def test_control_second_copy_of_a_registered_claim_fails(tiny):
    # The sentence is registered once and now appears twice.  Matching on
    # text as a set rather than counting copies would pass this.
    root, document = tiny
    path = _claims(root)
    path.write_text(path.read_text(encoding="utf-8")
                    + "A subnormal survives the multiply.\n",
                    encoding="utf-8")
    _git_add(root)
    problems = cc.check_census(root, document)
    assert any("unregistered claim" in problem
               for problem in problems), problems


def test_control_claim_moved_to_another_file_fails(tiny):
    root, document = tiny
    path = _claims(root)
    lines = path.read_text(encoding="utf-8").splitlines()
    moved = lines.pop(2)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "docs" / "other.md").write_text(moved + "\n", encoding="utf-8")
    _git_add(root)
    problems = cc.check_census(root, document)
    assert any(problem.startswith("unregistered claim docs/other.md:1")
               for problem in problems), problems
    assert any(problem.startswith(
        "registered claim no longer present docs/claims.md")
        for problem in problems), problems


def test_control_anchor_hash_edited_by_hand_fails(tiny):
    root, document = tiny
    document["sites"][0]["anchor_sha256"] = "0" * 64
    problems = cc.check_census(root, document)
    assert any("anchor hash stale" in problem
               for problem in problems), problems


def test_control_total_left_short_by_a_clean_merge_fails(tiny):
    # Two branches each register one claim and each write a total one above
    # their base.  Git takes two identical edits of the same line without a
    # conflict, so the merged register heads four records with a total of
    # three.
    root, document = tiny
    path = _claims(root)
    path.write_text(path.read_text(encoding="utf-8")
                    + "One branch adds an FTZ sentence.\n"
                    + "The other adds a subnormal sentence.\n",
                    encoding="utf-8")
    _git_add(root)
    merged = cc.build_census(root, document)
    assert cc.check_census(root, merged) == []
    merged["site_count"] = document["site_count"] + 1
    merged["site_count_by_kind"] = {"prose": document["site_count"] + 1}
    problems = cc.check_census(root, merged)
    assert any(problem.startswith("site_count says")
               for problem in problems), problems
    assert any(problem.startswith("site_count_by_kind says")
               for problem in problems), problems
