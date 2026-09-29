"""The resolved-scale disclaimer stays negated, and the guard can see it fail.

``tools/check_negation_invariant.py`` is only worth shipping if it changes
its verdict when the negation is removed.  So the essential tests here
are the three mutants: each deletes one negation from a scratch copy of
``README.md`` -- the mutants are built in ``tmp_path`` and never written
back to the tree -- and the checker must return FAIL where the unmutated
copy returns PASS.  A checker whose verdict does not move for any one of
them is non-discriminating for that failure mode and is not adopted.

M1 is the reason the guard enforces two scopes.  Its sentence still holds
an earlier negation ("not as a resolved-tornado claim"), so a purely
sentence-scoped checker keeps passing after the deletion; only the
line-scoped half sees it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tools.check_negation_invariant import check, documents, scan_text

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The published page that carries the negated claim the mutants edit.
#: The disclaimer lives in two places: VERIFICATION.md's "Not claimed"
#: section, which these mutants edit, and README.md's "Resolved scale."
#: item, whose three mutants the public receipt records
#: (tools/report_resolved_scale_disclaimer.py), so each page has a
#: discriminating mutant set of its own.
MUTANT_PAGE = Path("docs/public/VERIFICATION.md")

#: (name, exact source text, exact replacement) for each mutant.
MUTANTS = (
    (
        # The negation moves to the previous line: the sentence still
        # holds it, the line with the occurrence no longer does.
        "M1",
        "  case-study evidence, not a tornado-resolving claim.",
        "  case-study evidence, and not\n  a tornado-resolving claim.",
    ),
    (
        "M2",
        "- **No resolved tornado dynamics.**",
        "- **Resolved tornado dynamics.**",
    ),
    (
        "M3",
        "  case-study evidence, not a tornado-resolving claim.",
        "  case-study evidence, a tornado-resolving claim.",
    ),
)


def _scratch_tree(tmp_path: Path) -> Path:
    """Copy the scanned documents into a throwaway root."""
    root = tmp_path / "tree"
    (root / "docs" / "public").mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "README.md", root / "README.md")
    for page in (REPO_ROOT / "docs" / "public").glob("*.md"):
        shutil.copy2(page, root / "docs" / "public" / page.name)
    return root


def test_shipped_docs_negate_every_resolved_tornado_occurrence():
    report = check(REPO_ROOT)
    assert report["verdict"] == "PASS", report["occurrences"]
    # The disclaimer is the reason the pattern appears at all; if it ever
    # stops appearing, the guard has gone vacuous.
    assert report["occurrence_count"] > 0
    assert "README.md" in report["documents_scanned"]
    assert "docs/public/VERIFICATION.md" in report["documents_scanned"]


def test_scratch_copy_is_a_positive_control(tmp_path):
    """The copy itself must not move the verdict -- only a mutation may."""
    root = _scratch_tree(tmp_path)
    assert check(root)["verdict"] == "PASS"


@pytest.mark.parametrize("name,before,after", MUTANTS, ids=[m[0] for m in MUTANTS])
def test_deleting_one_negation_flips_the_verdict(tmp_path, name, before, after):
    root = _scratch_tree(tmp_path)
    page = root / MUTANT_PAGE
    text = page.read_text(encoding="utf-8")
    assert text.count(before) == 1, f"{name} anchor is not unique"
    page.write_text(text.replace(before, after), encoding="utf-8")

    report = check(root)
    assert report["verdict"] == "FAIL", name
    assert report["failure_count"] >= 1


def test_line_scope_is_what_catches_m1(tmp_path):
    """M1 keeps an earlier negation in its sentence; only the line sees it."""
    root = _scratch_tree(tmp_path)
    page = root / MUTANT_PAGE
    name, before, after = MUTANTS[0]
    page.write_text(
        page.read_text(encoding="utf-8").replace(before, after),
        encoding="utf-8",
    )
    failures = [
        item
        for item in check(root)["occurrences"]
        if not (item["negated_in_line"] and item["negated_in_sentence"])
    ]
    assert failures, name
    assert any(
        item["negated_in_sentence"] and not item["negated_in_line"]
        for item in failures
    ), failures


def test_guard_flags_an_affirmative_claim_in_any_published_page():
    """A new page making the claim outright fails, wherever it is written."""
    found = scan_text(
        "ArWen resolves tornado dynamics on the 500 m nest.\n", "docs/public/X.md"
    )
    assert len(found) == 1
    assert not found[0].ok


def test_documents_scan_readme_and_every_public_page():
    scanned = {path.relative_to(REPO_ROOT).as_posix() for path in documents(REPO_ROOT)}
    published = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "docs" / "public").glob("*.md")
    }
    assert published
    assert published | {"README.md"} == scanned
