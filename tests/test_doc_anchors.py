"""Documentation line citations must still point at what they claim.

THE BREAKAGE THIS PREVENTS, measured on this tree 2026-09-17.
``docs/manual/02-dynamics-grids-nesting.md`` sent a reader to
``docs/public/CONFIGURATION.md:235`` and ``:414`` for the ``scalar_adv_opt``
matching rule.  :235 is the middle of a bare list of other key names and
:414 is a sentence about sealed preparation states; the rows that carry the
rule are :332 and :627.  Two lines lower it sent the same reader to :90 for
``spec_bdy_width``, which is the ``## Tweakable knobs`` heading, not the row
at :103.  Three more were the same shape: ``rk_ord`` at :406 instead of
:619, ``h_sca_adv_order`` at :234 instead of :331, and
``radt_ladder_minutes`` at ``gpuwm/domain_wizard.py:1754-1786`` when the
function had moved to :1931-1963.  Nothing checked any of them, and a
citation that lands on unrelated text leaves a reader unable to tell whether
the claim or the pointer is wrong.

This file is the standing half: ``tools/check_doc_anchors.py`` resolves every
in-scope citation, and the first test holds the count of offences at zero.
The rest pin the three judgements the checker makes, because each was wrong
once during the measurement and a checker that mis-classifies is worse than
none: it cries wolf until somebody deletes it.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re

import pytest

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKER_PATH = REPOSITORY_ROOT / "tools" / "check_doc_anchors.py"


def _checker():
    """Load the checker BY PATH.

    ``tools`` is importable in a development checkout and is not in a
    published snapshot, and a gate that silently skips when its helper
    cannot be imported is not a gate.  A missing file is a failure with the
    path in it.
    """

    assert CHECKER_PATH.is_file(), (
        f"{CHECKER_PATH} is missing; the documentation anchors then have no "
        "checker and every citation below goes unread")
    specification = importlib.util.spec_from_file_location(
        "gpuwm_doc_anchor_checker", CHECKER_PATH)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


CHECKER = _checker()


def test_no_documentation_anchor_has_drifted() -> None:
    """The property: every resolvable in-scope citation lands on its anchor."""

    result = CHECKER.survey()
    assert not result["drifted"] and not result["unresolvable"], (
        "these documentation citations no longer point at the key or symbol "
        "their own sentence names.  Re-point the citation at the line the "
        "checker reports, or fix the sentence if the claim moved:\n  "
        + "\n  ".join(result["drifted"] + result["unresolvable"]))


def test_the_survey_actually_resolved_something() -> None:
    """Guard the guard: a survey that checks nothing passes vacuously.

    The scope is narrow by design -- the configuration reference and the
    package's own modules -- so the count is small, and a change that made
    the citation pattern or the scope test stop matching would turn the
    test above green while reading nothing at all.
    """

    result = CHECKER.survey()
    assert result["checked"] >= 8, (
        f"the anchor survey resolved only {result['checked']} citations; it "
        "resolved 11 when this floor was recorded, so either the citation "
        "pattern or the in-scope test has stopped matching and the gate "
        "above is passing on an empty set")


def test_only_an_identifier_shaped_name_is_taken_as_an_anchor() -> None:
    """The bar that separates an instrument from a noise generator.

    Without it the survey read ordinary backticked English -- ``and``,
    ``then``, ``flag``, ``rust``, ``required`` -- as if each were a symbol,
    and reported 41 of 48 citations as drifted on anchors their sentences
    never meant.
    """

    prose = ("The importer pins `scalar_adv_opt` and `QNICE`, `wrfinput_d01` "
             "`and` `then` `rust` `flag` `x` [docs/public/CONFIGURATION.md:1].")
    match = CHECKER._CITATION.search(prose)
    assert match is not None
    names = CHECKER.anchor_names(prose, match)
    assert "scalar_adv_opt" in names and "QNICE" in names
    assert "wrfinput_d01" in names
    for noise in ("and", "then", "rust", "flag", "x"):
        assert noise not in names, (
            f"{noise!r} was taken as an anchor; ordinary backticked English "
            "is not a symbol and reading it as one is how this checker "
            "reported 41 false findings")


def test_a_document_that_names_its_revision_is_resolved_there() -> None:
    """A receipt's citations are evidence at the revision it states.

    Resolving them against HEAD reports drift that is not drift, and the
    only way to make such a report green would be to re-point a receipt at
    lines its run never read -- which falsifies the receipt.  Measured: the
    observation battery's B4 receipt says "against `fc15d9ae`" and four of
    its citations are correct there and stale against HEAD.
    """

    receipt = (REPOSITORY_ROOT / "docs" / "public" / "receipts" / "obsbattery"
               / "B4-ROUTE-QUALIFICATION.md")
    if not receipt.is_file():
        pytest.skip("the observation battery's B4 receipt is not in this tree")
    revision = CHECKER.revision_of(receipt.read_text(encoding="utf-8"))
    assert revision is not None and re.fullmatch(r"[0-9a-f]{7,40}", revision), (
        "the B4 receipt states the revision it was written against in its "
        "opening lines, and the checker must read it; without that its "
        "citations are judged against a tree they were never taken from")
    assert CHECKER.survey()["pinned"] > 0, (
        "no citation was resolved against a document's own revision, so the "
        "pinned-revision route is dead code and the receipts are being "
        "judged against HEAD")


def test_a_quoted_phrase_outranks_a_symbol_name() -> None:
    """A phrase quoted out of the cited lines is the strongest anchor.

    And its extraction has to survive an inline ``key="value"`` in the same
    paragraph: a pair-by-pair scan took that value's closing mark as an
    opening one, swallowed the rest of the sentence and never saw the quote
    the citation was anchored on -- which is how a correct receipt citation
    was reported as drifted.
    """

    prose = ('The gate is catchable, `stock_wrf_export="optional"`, and\n'
             '`gpuwm/native_hierarchy.py:1-2` says so in its own words: '
             '"Nothing about the gate\'s CONTENT changes here."')
    match = CHECKER._CITATION.search(prose)
    assert match is not None
    quotes = CHECKER.anchor_quotes(prose, match)
    assert "Nothing about the gate's CONTENT changes here" in quotes, (
        f"the quoted phrase was not extracted (got {quotes!r}); the inline "
        "assignment's quotation marks have broken the pairing again")
