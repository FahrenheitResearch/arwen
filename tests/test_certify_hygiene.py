"""The certification path stays generic, and the docs stay derived.

F3-AC12: no case identifier reaches ``gpuwm/certify/**``, the capsule schema,
or any generic identifier, and nothing on the certification import path pulls
in a case module.  The specific temptation is real and named: the case chain
module aliases a sibling case module as ``shared`` and takes ``stable_hash``
from it, so borrowing that helper would put a case module one import away from
every capsule.

F3-AC13: ``docs/public/`` never says "liveness heartbeat", and a statement of
how many pins a receipt covers is counted from the receipt's own key list
rather than typed as a number that stops being true.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CERTIFY = REPO / "gpuwm" / "certify"
DOCS_PUBLIC = REPO / "docs" / "public"

#: Case identifiers that must never appear in generic code or schemas.
CASE_TOKENS = ("real74", "1974", "ohio", "hrrr")


def _certify_files() -> list[Path]:
    return sorted(p for p in CERTIFY.rglob("*")
                  if p.is_file() and p.suffix in {".py", ".json"})


def test_the_certify_package_exists_and_this_test_is_not_vacuous():
    files = _certify_files()
    assert len(files) >= 5, files
    assert (CERTIFY / "capsule_v1.schema.json") in files


@pytest.mark.parametrize("token", CASE_TOKENS)
def test_no_case_identifier_appears_anywhere_under_certify(token):
    offenders = []
    for path in _certify_files():
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            if token in line.lower():
                offenders.append(f"{path.relative_to(REPO)}:{number}: {line}")
    assert offenders == [], offenders


def test_no_case_identifier_appears_in_the_capsule_schema_key_set():
    schema = (CERTIFY / "capsule_v1.schema.json").read_text(encoding="utf-8")
    keys = set(re.findall(r'"([A-Za-z0-9_.:/+-]+)"\s*:', schema))
    for key in keys:
        assert not any(token in key.lower() for token in CASE_TOKENS), key


def test_nothing_on_the_certification_import_path_imports_a_case_module():
    """The named temptation: a case module reached through a generic helper."""
    offenders = []
    for path in _certify_files():
        if path.suffix != ".py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                module = node.module
            elif isinstance(node, ast.Import):
                module = ",".join(alias.name for alias in node.names)
            else:
                continue
            if "verify.cases" in module or any(
                    token in module.lower() for token in CASE_TOKENS):
                offenders.append(f"{path.relative_to(REPO)}: {module}")
    assert offenders == [], offenders


def test_the_case_helper_this_criterion_names_is_still_where_it_says():
    """Grounding: the alias and the helper the criterion warns about exist."""
    chain = REPO / "gpuwm" / "verify" / "cases" / "real74_chain.py"
    sibling = REPO / "gpuwm" / "verify" / "cases" / "real74_d02.py"
    assert "as shared" in chain.read_text(encoding="utf-8")
    assert "def stable_hash" in sibling.read_text(encoding="utf-8")
    # And no module under gpuwm/certify reaches for it.
    for path in _certify_files():
        assert "stable_hash" not in path.read_text(encoding="utf-8")


def test_the_certification_band_is_addressed_by_config_identity():
    """[D-18] the band is keyed by config identity, never by a case literal."""
    from gpuwm.certify.pins import PINS

    assert any(pin.key == "config_bytes" for pin in PINS)
    for pin in PINS:
        assert not any(token in pin.key.lower() for token in CASE_TOKENS)


# --- F3-AC13 ---------------------------------------------------------------

def _public_docs() -> list[Path]:
    return sorted(p for p in DOCS_PUBLIC.rglob("*.md") if p.is_file())


def test_the_phrase_liveness_heartbeat_appears_nowhere_under_docs_public():
    offenders = []
    for path in _public_docs():
        for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1):
            if "liveness heartbeat" in line.lower():
                offenders.append(f"{path.relative_to(REPO)}:{number}")
    assert offenders == [], offenders


def test_no_public_document_states_a_pin_count_as_a_literal():
    """A typed count stops being true the moment the pin table moves."""
    # The negative lookbehind keeps a HYPHENATED number out of the
    # match: "SHA-256 pins" names a hash beside the word, it does not
    # state a count of pins, and reading it as one made this rule fire
    # on a sentence that states no count at all.  A real count ("five
    # pins", "12 pins") never has a hyphen in front of the number.
    pattern = re.compile(
        r"(?<!-)\b(one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|thirteen|\d+)\s+(of\s+the\s+)?pins?\b",
        re.IGNORECASE)
    offenders = []
    for path in _public_docs():
        for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(REPO)}:{number}: {line}")
    assert offenders == [], offenders


def test_the_failure_capsule_pin_coverage_is_counted_from_its_key_list():
    """Five pin-table rows, counted -- the 'gpu' field alone spans two."""
    from gpuwm.certify.pins import PINS

    # The payload fields of gpuwm.supervisor.write_failure_capsule that carry
    # pinned identity, mapped to the pin items each one records.
    payload_coverage = {
        "config_sha256": ("config_bytes",),
        "input_hashes": ("input_artifact_bytes",),
        "git_commit": ("arwen_version_and_commit",),
        "gpu": ("gpu_identity", "cuda_driver_version"),
    }
    by_key = {pin.key: pin for pin in PINS}
    covered_items = {key for keys in payload_coverage.values() for key in keys}
    covered_rows = {by_key[key].doc_row for key in covered_items}
    assert len(covered_items) == len(covered_rows) == 5, (
        "the count is derived from the payload's key list; if a field is "
        "added or a pin row splits, this number moves with it")
    assert covered_items < set(by_key)
    assert len(by_key) - len(covered_items) == 8


def test_the_determinism_pin_table_is_the_source_the_module_quotes():
    from gpuwm.certify.pins import PINS, PIN_TABLE_DOC

    text = (REPO / PIN_TABLE_DOC).read_text(encoding="utf-8")
    for pin in PINS:
        assert pin.doc_row in text, pin.key
