"""``gpuwm certify``: every refusal, and a verdict that binds itself.

The refusal tests run the shipped command through :func:`gpuwm.cli.main`, so
what is asserted is the exit code and the message a reader actually gets, not
the return of a helper the command might not call.

Nothing here transcribes a published or measured number.  Every metrics value
is derived from the interval it is tested against, so an interval that moved
moves its own fixture with it and no test can pre-ordain a pass.
"""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

import certification_fixtures as fixtures
from gpuwm.certify import verdict as verdict_module
from gpuwm.certify.verdict import (BOUND_KEYS, CONDITIONS, bound_inventory,
                                   capsule_binding_sha256, certify,
                                   rederive_verdict, verify_verdict)
from gpuwm.cli import main

REPO_ROOT = fixtures.REPO_ROOT


def _run_certify(paths, out_verdict: Path | None = None) -> int:
    argv = ["certify",
            "--run-capsule", str(paths["capsule"]),
            "--metrics-csv", str(paths["metrics"]),
            "--band", str(paths["band"]),
            "--wrf-reference-manifest", str(paths["wrf_reference"])]
    if out_verdict is not None:
        argv += ["--out-verdict", str(out_verdict)]
    return main(argv)


def _verdict_for(paths) -> dict:
    return certify(capsule_path=paths["capsule"],
                   metrics_csv=paths["metrics"],
                   band_path=paths["band"],
                   wrf_reference_manifest=paths["wrf_reference"])


# --------------------------------------------------------------------------
# AC4 -- the matched fixture passes, and each refusal fires on its own
# --------------------------------------------------------------------------

def test_the_matched_fixture_certifies(tmp_path, capsys):
    paths = fixtures.matched_set(tmp_path)
    out = tmp_path / "verdict.json"
    assert _run_certify(paths, out) == 0
    assert "certify: PASS" in capsys.readouterr().out
    document = json.loads(out.read_text(encoding="utf-8"))
    assert document["passed"] is True
    assert {item["condition"] for item in document["conditions"]} == set(
        CONDITIONS)
    assert document["comparisons"], "a passing verdict bound no comparison"


def _refusal(tmp_path, capsys, **kwargs) -> str:
    paths = fixtures.matched_set(tmp_path, **kwargs)
    assert _run_certify(paths) != 0
    return capsys.readouterr().err


def test_refuses_a_band_keyed_to_another_configuration(tmp_path, capsys):
    def rekey(band):
        band["config_sha256"] = "1" * 64

    message = _refusal(tmp_path, capsys, band_edit=rekey)
    assert "band_config_identity_matches_capsule" in message


@pytest.mark.parametrize("hash_key", [
    "wrf_exe_sha256", "build_recipe_sha256", "namelist_sha256",
    "reference_wrfout_sha256"])
def test_refuses_an_absent_wrf_reference_hash(tmp_path, capsys, hash_key):
    def drop(reference):
        reference[hash_key] = None

    message = _refusal(tmp_path, capsys, reference_edit=drop)
    assert "wrf_reference_hashes_present" in message
    assert hash_key in message


def test_refuses_inventory_mode_geography(tmp_path, capsys):
    def to_inventory(capsule):
        capsule["input_bytes"]["entries"]["geography_root"]["algorithm"] = (
            "sha256-directory-inventory")

    message = _refusal(tmp_path, capsys, capsule_edit=to_inventory)
    assert "geography_input_hashed_by_content" in message
    assert "geography_root" in message


def test_refuses_a_pin_that_is_not_resolved(tmp_path, capsys):
    def unresolve(capsule):
        capsule["numerical_stack"]["numpy_version"] = {
            "value": None, "source": "importlib.metadata",
            "status": "unavailable", "reason": "fixture"}

    message = _refusal(tmp_path, capsys, capsule_edit=unresolve)
    assert "every_pin_resolved" in message
    assert "numpy_version" in message


def test_refuses_a_metrics_column_the_band_does_not_classify(tmp_path, capsys):
    message = _refusal(tmp_path, capsys,
                       extra_columns={"theta_e_mae": "0.25"})
    assert "every_metrics_column_classified" in message
    assert "theta_e_mae" in message


def test_refuses_a_banded_row_outside_its_own_interval(tmp_path, capsys):
    message = _refusal(tmp_path, capsys, out_of_band=True)
    assert "every_banded_row_inside_its_interval" in message


def test_refuses_a_capsule_that_does_not_validate(tmp_path, capsys):
    def cripple(capsule):
        del capsule["kernel_manifest"]

    message = _refusal(tmp_path, capsys, capsule_edit=cripple)
    assert "capsule_validates" in message


# --------------------------------------------------------------------------
# The empty comparison.  `certify` exited 0 on a metrics CSV with no data
# rows -- "certify: PASS (0 banded comparisons ...)" -- because every row
# condition is a universal quantifier and a universal quantifier over
# nothing is true.  Both degenerate shapes are tested, and so is the
# legitimate one, because a guard that fires on a real comparison is
# worse than the hole it closes.
# --------------------------------------------------------------------------

def test_refuses_a_metrics_csv_with_a_header_and_no_data_rows(
        tmp_path, capsys):
    paths = fixtures.matched_set(tmp_path)
    band = fixtures.shipped_band()
    columns, rows = fixtures.metrics_rows(band)
    assert rows, "the fixture produced no rows, so this test proves nothing"
    fixtures.write_metrics_csv(paths["metrics"], columns, [])
    assert _run_certify(paths) != 0
    message = capsys.readouterr().err
    assert "the_comparison_is_not_empty" in message
    # The refusal states what it found and what it required.
    assert "0 data row(s)" in message
    assert "0 banded comparison(s)" in message
    assert "at least 1 row and 1 banded comparison are required" in message


def test_refuses_metrics_rows_the_band_gates_none_of(tmp_path, capsys):
    """All-skipped: real rows, no banded column among them.

    Reached with the shipped band unedited -- a CSV carrying its row keys
    and no metric column.  Every declared column is classified (there are
    none), so the pre-existing coverage condition passes and the run
    would have certified on rows nothing was measured from.
    """

    from gpuwm.certify.band import ROW_KEY_COLUMNS

    paths = fixtures.matched_set(tmp_path)
    band = fixtures.shipped_band()
    _columns, rows = fixtures.metrics_rows(band)
    keys_only = [{key: row[key] for key in ROW_KEY_COLUMNS} for row in rows]
    fixtures.write_metrics_csv(paths["metrics"], list(ROW_KEY_COLUMNS),
                               keys_only)
    assert _run_certify(paths) != 0
    message = capsys.readouterr().err
    assert "the_comparison_is_not_empty" in message
    assert f"{len(keys_only)} data row(s)" in message
    assert "the band gates 0 of its 3 column(s)" in message


def test_the_empty_comparison_guard_does_not_fire_on_a_real_comparison(
        tmp_path, capsys):
    """The negative half: the matched fixture still certifies.

    Named separately from ``test_the_matched_fixture_certifies`` because
    this one asserts the new condition specifically -- reported,
    satisfied, and carrying the counts -- so a future edit that made the
    guard over-eager fails here with the reason visible.
    """

    paths = fixtures.matched_set(tmp_path)
    document = _verdict_for(paths)
    condition = next(item for item in document["conditions"]
                     if item["condition"] == "the_comparison_is_not_empty")
    assert condition["satisfied"] is True, condition
    assert f"{document['metrics']['row_count']} metrics rows" in \
        condition["detail"]
    assert f"= {len(document['comparisons'])} banded comparisons" in \
        condition["detail"]
    assert _run_certify(paths) == 0
    assert "certify: PASS" in capsys.readouterr().out


def test_a_verdict_carrying_no_comparison_cannot_rederive_as_a_pass(tmp_path):
    """Belt and braces, at the independent-verifier door.

    ``rederive_verdict`` recomputes the decision from the bound rows
    alone, and every clause it applies to comparisons is vacuous on an
    empty list.  Emptying the list must not turn a verdict into a pass.
    """

    paths = fixtures.matched_set(tmp_path)
    document = _verdict_for(paths)
    assert rederive_verdict(document) is True
    hollow = copy.deepcopy(document)
    hollow["comparisons"] = []
    for item in hollow["conditions"]:
        item["satisfied"] = True
    assert rederive_verdict(hollow) is False


def test_every_declared_condition_is_reported_whether_or_not_it_fired(
        tmp_path):
    paths = fixtures.matched_set(tmp_path, out_of_band=True)
    document = _verdict_for(paths)
    reported = [item["condition"] for item in document["conditions"]]
    assert reported == list(CONDITIONS)
    assert document["passed"] is False


# --------------------------------------------------------------------------
# AC6 -- the verdict binds itself, and the claimed bit is inert
# --------------------------------------------------------------------------

def test_the_binding_covers_the_closed_inventory_without_passed(tmp_path):
    paths = fixtures.matched_set(tmp_path)
    document = _verdict_for(paths)
    inventory = bound_inventory(document)
    assert "passed" not in inventory
    assert "capsule_binding_sha256" not in inventory
    assert set(inventory) == set(BOUND_KEYS)
    # Everything the decision rests on, and nothing else, is bound.
    assert set(document) == set(BOUND_KEYS) | {"passed",
                                               "capsule_binding_sha256"}
    intact, rederived = verify_verdict(document)
    assert intact and rederived is document["passed"]


def test_flipping_or_deleting_passed_is_inert(tmp_path):
    paths = fixtures.matched_set(tmp_path)
    document = _verdict_for(paths)
    baseline_hash = document["capsule_binding_sha256"]
    baseline_verdict = rederive_verdict(document)

    flipped = copy.deepcopy(document)
    flipped["passed"] = not flipped["passed"]
    assert capsule_binding_sha256(flipped) == baseline_hash
    assert rederive_verdict(flipped) is baseline_verdict

    deleted = copy.deepcopy(document)
    del deleted["passed"]
    assert capsule_binding_sha256(deleted) == baseline_hash
    assert rederive_verdict(deleted) is baseline_verdict


def _break_condition(document, index):
    document["conditions"][index]["satisfied"] = False


def _break_comparison(document, index):
    """Push one bound value one interval width past its own upper endpoint."""
    row = document["comparisons"][index]
    if row["nan_expected"]:
        row["value"] = 0.0
        return
    width = float(row["upper"]) - float(row["lower"])
    row["value"] = float(row["upper"]) + (width if width > 0.0 else 1.0)


def test_mutating_any_bound_row_moves_both_the_hash_and_the_verdict(tmp_path):
    paths = fixtures.matched_set(tmp_path)
    document = _verdict_for(paths)
    assert document["passed"] is True
    baseline_hash = document["capsule_binding_sha256"]

    mutated_rows = 0
    for index in range(len(document["conditions"])):
        candidate = copy.deepcopy(document)
        _break_condition(candidate, index)
        assert capsule_binding_sha256(candidate) != baseline_hash
        assert rederive_verdict(candidate) is False
        mutated_rows += 1
    for index in range(len(document["comparisons"])):
        candidate = copy.deepcopy(document)
        _break_comparison(candidate, index)
        assert capsule_binding_sha256(candidate) != baseline_hash
        assert rederive_verdict(candidate) is False
        mutated_rows += 1
    assert mutated_rows == (len(document["conditions"])
                            + len(document["comparisons"]))
    assert mutated_rows > len(CONDITIONS), "no comparison row was bound"


def test_relabelling_a_bound_row_still_moves_the_hash(tmp_path):
    """A label-only edit cannot be laundered past the binding."""
    paths = fixtures.matched_set(tmp_path)
    document = _verdict_for(paths)
    baseline_hash = document["capsule_binding_sha256"]
    for key in BOUND_KEYS:
        candidate = copy.deepcopy(document)
        if isinstance(candidate[key], list):
            candidate[key].append({"injected": True})
        elif isinstance(candidate[key], dict):
            candidate[key]["injected"] = True
        else:
            candidate[key] = f"{candidate[key]}-mutated"
        assert capsule_binding_sha256(candidate) != baseline_hash, key


def test_a_recorded_inside_that_disagrees_with_its_own_bounds_fails(tmp_path):
    paths = fixtures.matched_set(tmp_path, out_of_band=True)
    document = _verdict_for(paths)
    laundered = copy.deepcopy(document)
    for row in laundered["comparisons"]:
        row["inside"] = True
        row["placement"] = "inside"
    for item in laundered["conditions"]:
        item["satisfied"] = True
    assert rederive_verdict(laundered) is False


# --------------------------------------------------------------------------
# The verdict schema version, and reading a verdict written before it moved
# --------------------------------------------------------------------------

def _as_previous_schema(document: dict) -> dict:
    """The same verdict as base 1.0.0 would have written it.

    1.0.0 declared seven conditions; ``the_comparison_is_not_empty`` is the
    eighth.  Dropping that row and stamping the older version is exactly
    the document a run certified before this branch produced -- verified
    against a real one: a verdict written by the base commit carries these
    seven condition names in this order and ``verdict_schema_version``
    ``1.0.0``.  The binding is recomputed because a genuine older verdict
    bound its own contents, not today's.
    """

    older = copy.deepcopy(document)
    older["verdict_schema_version"] = "1.0.0"
    older["conditions"] = [item for item in older["conditions"]
                           if item["condition"] != "the_comparison_is_not_empty"]
    older["capsule_binding_sha256"] = capsule_binding_sha256(older)
    return older


def test_todays_conditions_are_the_set_todays_schema_version_declares():
    from gpuwm.certify.verdict import (CONDITIONS_BY_SCHEMA_VERSION,
                                       VERDICT_SCHEMA_VERSION)

    assert CONDITIONS_BY_SCHEMA_VERSION[VERDICT_SCHEMA_VERSION] == CONDITIONS
    # The version moved because the set moved: no two published versions
    # may declare the same conditions, or the version says nothing.
    sets = [frozenset(value)
            for value in CONDITIONS_BY_SCHEMA_VERSION.values()]
    assert len(set(sets)) == len(sets), CONDITIONS_BY_SCHEMA_VERSION


def test_a_verdict_from_the_previous_schema_version_still_rederives(tmp_path):
    """A genuine older verdict is not a forgery, and must not read as one.

    ``rederive_verdict`` compares the condition set for exact equality.
    Adding the eighth condition without moving the version made every
    verdict written by the previous release rederive ``False`` -- the same
    silent answer a tampered document gets, with no signal to tell an
    independent verifier which they were holding.
    """

    from gpuwm.certify.verdict import rederive_verdict_reason

    document = _verdict_for(fixtures.matched_set(tmp_path))
    assert document["verdict_schema_version"] == "1.1.0"
    assert rederive_verdict(document) is True

    older = _as_previous_schema(document)
    passed, why = rederive_verdict_reason(older)
    assert passed is True, why
    assert "1.0.0" in why
    intact, rederived = verify_verdict(older)
    assert intact and rederived is True


def test_dropping_a_condition_without_the_version_is_still_a_forgery(tmp_path):
    """The other direction: the version is what makes the older set legal.

    Same document, same missing row -- but still claiming 1.1.0.  This is
    the tamper case the exact-set comparison exists for, and it must stay
    red, or the read path for old verdicts has become a hole.
    """

    from gpuwm.certify.verdict import rederive_verdict_reason

    document = _verdict_for(fixtures.matched_set(tmp_path))
    forged = _as_previous_schema(document)
    forged["verdict_schema_version"] = "1.1.0"
    forged["capsule_binding_sha256"] = capsule_binding_sha256(forged)
    passed, why = rederive_verdict_reason(forged)
    assert passed is False
    assert "the_comparison_is_not_empty" in why
    assert "1.1.0" in why


def test_an_unknown_verdict_schema_version_fails_closed_by_name(tmp_path):
    from gpuwm.certify.verdict import rederive_verdict_reason

    document = _verdict_for(fixtures.matched_set(tmp_path))
    for claimed in ("9.9.9", "", None, 1.0):
        candidate = copy.deepcopy(document)
        candidate["verdict_schema_version"] = claimed
        passed, why = rederive_verdict_reason(candidate)
        assert passed is False, claimed
        assert "verdict_schema_version" in why and "1.1.0" in why, why


def test_reading_the_older_set_does_not_reopen_the_empty_comparison(tmp_path):
    """The floor is version-independent, so 1.0.0 cannot be green on nothing.

    A 1.0.0 verdict never declared ``the_comparison_is_not_empty``.  If the
    read path simply honoured its condition list, an empty older verdict
    would rederive as a pass -- the exact hole the eighth condition closed,
    reached through the compatibility door.
    """

    from gpuwm.certify.verdict import rederive_verdict_reason

    older = _as_previous_schema(_verdict_for(fixtures.matched_set(tmp_path)))
    assert rederive_verdict(older) is True
    hollow = copy.deepcopy(older)
    hollow["comparisons"] = []
    for item in hollow["conditions"]:
        item["satisfied"] = True
    hollow["capsule_binding_sha256"] = capsule_binding_sha256(hollow)
    passed, why = rederive_verdict_reason(hollow)
    assert passed is False
    assert "0 comparison" in why


# --------------------------------------------------------------------------
# The public reference document says what the code declares
# --------------------------------------------------------------------------

def _documented_conditions() -> list[str]:
    """The condition names in CERTIFICATION.md's table, in document order."""
    text = (REPO_ROOT / "docs" / "public"
            / "CERTIFICATION.md").read_text(encoding="utf-8")
    names: list[str] = []
    inside = False
    for line in text.splitlines():
        if line.startswith("| Condition |"):
            inside = True
            continue
        if inside:
            if not line.startswith("|"):
                break
            cell = line.split("|")[1].strip()
            if cell.startswith("`") and cell.endswith("`"):
                names.append(cell.strip("`"))
    return names


def test_the_public_doc_lists_every_condition_the_code_declares():
    """CERTIFICATION.md presents its table as complete; hold it to that.

    The doc says "It returns 0 only when every condition below holds" and
    then lists them.  When the eighth condition landed, the table stayed at
    seven and nothing was red, because nothing tied the public reference
    for this command to :data:`CONDITIONS`.  Now something does, in order,
    so a condition added or renamed has to visit the document.
    """

    assert _documented_conditions() == list(CONDITIONS)


# --------------------------------------------------------------------------
# AC6 -- the certification import path carries no case module
# --------------------------------------------------------------------------

def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module)
    return names


def test_nothing_under_certify_imports_a_case_module():
    sources = sorted((REPO_ROOT / "gpuwm" / "certify").rglob("*.py"))
    assert sources, "gpuwm/certify carries no modules to check"
    offenders = {
        path.name: sorted(name for name in _imported_modules(path)
                          if name.startswith("gpuwm.verify"))
        for path in sources
    }
    assert not any(offenders.values()), offenders


def test_the_temptation_the_criterion_names_is_still_there_to_resist():
    """The helper certify would have reused lives on a case module.

    If this ever fails, the import-graph test above may have gone vacuous --
    it would be asserting the absence of something that no longer exists.
    """
    chain = (REPO_ROOT / "gpuwm" / "verify" / "cases"
             / "real74_chain.py").read_text(encoding="utf-8")
    assert "as shared" in chain
    assert "_controller_score_binding_sha256" in chain
    assert "def stable_hash(" in (
        REPO_ROOT / "gpuwm" / "verify" / "cases"
        / "real74_d02.py").read_text(encoding="utf-8")
    # certify mirrors the shape and owns its own digest, under its own name.
    assert "def canonical_digest(" in (
        REPO_ROOT / "gpuwm" / "certify"
        / "verdict.py").read_text(encoding="utf-8")
    assert verdict_module.canonical_digest({"b": 1, "a": 2}) == (
        verdict_module.canonical_digest({"a": 2, "b": 1}))


# --------------------------------------------------------------------------
# The four file arguments name themselves when they cannot be read
# (1.8.8 refusal sweep, finding 2)
# --------------------------------------------------------------------------

#: The flag, and the phrase the reader uses for what that file is.
CERTIFY_FILE_ARGUMENTS = [
    ("--run-capsule", "capsule", "a certification capsule"),
    ("--metrics-csv", "metrics", "a matched-comparison metrics table"),
    ("--band", "band", "an acceptance band"),
    ("--wrf-reference-manifest", "wrf_reference",
     "a WRF reference manifest"),
]


@pytest.mark.parametrize("flag,key,what", CERTIFY_FILE_ARGUMENTS)
def test_a_zero_byte_argument_is_refused_by_its_own_name(
        tmp_path, capsys, flag, key, what):
    """Which of the four is unreadable is the load-bearing part.

    `gpuwm certify` used to read all four inputs with a bare
    ``read_bytes`` + ``json.loads``, so ANY of them being empty produced
    the same six words -- ``gpuwm certify: Expecting value: line 1
    column 1 (char 0)`` -- with no file, no argument, no byte count and
    no way through.  `gpuwm dual-run` had answered the identical file
    with arm, path, byte count and remedy since 1.8.8; the fix had
    landed on one command and not its sibling.

    Parametrized over all four arguments on purpose: a reader that names
    only the capsule would pass a single-argument test and still strand
    whoever pointed --band at the wrong file.
    """

    paths = fixtures.matched_set(tmp_path)
    paths[key].write_bytes(b"")

    assert _run_certify(paths) == 2
    message = capsys.readouterr().err
    assert flag in message, message
    assert str(paths[key]) in message, message
    assert "is empty (0 bytes)" in message, message
    assert f"That is not {what}" in message, message
    # A way through, not just a diagnosis.
    assert f"Point {flag} at" in message, message


@pytest.mark.parametrize("flag,key,what", [
    row for row in CERTIFY_FILE_ARGUMENTS if row[0] != "--metrics-csv"])
def test_a_malformed_json_argument_is_refused_by_its_own_name(
        tmp_path, capsys, flag, key, what):
    """Same for a file that has bytes but is not JSON.

    The metrics CSV is excluded because it is not JSON: its emptiness is
    guarded above and its shape is guarded by the header/row conditions.
    """

    paths = fixtures.matched_set(tmp_path)
    paths[key].write_text("{ this is not json\n", encoding="utf-8")

    assert _run_certify(paths) == 2
    message = capsys.readouterr().err
    assert flag in message, message
    assert str(paths[key]) in message, message
    assert "is not valid JSON" in message, message
    # The parser's own coordinates, so a long document can be opened at
    # the right place rather than re-read from the top.
    assert "at line 1 column" in message, message
    assert f"Point {flag} at" in message, message


def test_the_reader_covers_every_file_argument_the_parser_declares():
    """The table and the parser cannot drift apart.

    A fifth file argument added to `gpuwm certify` without an entry in
    CERTIFY_INPUTS would fall back to the generic wording, which is the
    silent half-fix this finding was about.  Read off the real parser.
    """

    import argparse

    from gpuwm.certify.cli import register_cli
    from gpuwm.certify.verdict import CERTIFY_INPUTS

    parser = argparse.ArgumentParser()
    register_cli(parser.add_subparsers(dest="command"))
    certify_parser = parser._subparsers._group_actions[0].choices["certify"]
    inputs = {
        action.option_strings[0] for action in certify_parser._actions
        if action.option_strings and action.type is Path
        and action.required}

    assert inputs == set(CERTIFY_INPUTS), (
        "gpuwm certify's required file arguments and the shared reader's "
        f"table disagree: parser {sorted(inputs)}, table "
        f"{sorted(CERTIFY_INPUTS)}")
