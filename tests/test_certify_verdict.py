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
