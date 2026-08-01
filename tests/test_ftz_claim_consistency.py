"""The FTZ claim census covers the tree, and it can be made to fail.

Three mutations drive the three ways a census goes stale: a claim sentence
nobody registered, a token that contradicts the receipt, and an anchor whose
quoted text was edited underneath the record.  A census gate that survives
any of them is decoration.
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
