"""The top CHANGELOG section is a release note, not a development log.

1.4.1 shipped 5,910 words of release notes carrying internal development state:
branch names, an internal `Verdict: BLOCKED`, a roadmap target for the next
version, and the rebase status of unshipped work. For scale, 1.2.3 shipped 94
words, 1.3.1 shipped 853 and 1.3.0 shipped 1,252. None of it was secret, but
none of it was a user's business either, and a reader cannot tell a shipped
capability from a branch name.

Two failure modes, and they are different in kind:

* **Leaking internal state** is unambiguous -- a branch prefix or an internal
  verdict either appears or it does not -- so it is a hard failure.
* **Length** is a judgement call, so the bound here is set where no reasonable
  release note lands rather than at what a good one looks like. It exists to
  catch a development log pasted wholesale, not to referee prose.

Only the NEWEST section is checked. History is history: earlier entries are not
rewritten to satisfy a rule written after they shipped.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CHANGELOG = REPO / "CHANGELOG.md"

#: Well above any release note this project has shipped (largest before 1.4.0
#: was 1,252 words) and far below a pasted development log (5,910). A release
#: that genuinely needs more prose should link to a document, not inline it.
MAX_WORDS = 3500  # project ruling, 2026-09-21: raised from 3000 for 2.7.6's thirty-nine changes

#: A release given more room than MAX_WORDS, by version, with the ruling that
#: gave it.  Every release not named here keeps MAX_WORDS.
#:
#: 2.8.0: owner ruling, 2026-09-28, verbatim: "word count can be 7k for this
#: one since its a huge release".  Condensed to 3,500 words, the 2.8.0 section
#: had to drop script endpoints, warning codes and settings a user acts on.
RELEASE_WORD_ALLOWANCE: dict[str, int] = {"2.8.0": 7000}


def word_limit(heading: str) -> int:
    """The word limit for the release a ``## <version> ...`` heading names."""

    found = re.match(r"^##\s+(\S+)", heading)
    version = found.group(1) if found else ""
    return RELEASE_WORD_ALLOWANCE.get(version, MAX_WORDS)


#: Substrings that mean internal development state reached a public document.
#: Each is paired with what it leaks, so a failure explains itself.
LEAKS: tuple[tuple[str, str], ...] = (
    ("lane/", "a branch name"),
    ("feature/", "a branch name"),
    ("integration/", "a branch name"),
    ("product/v", "a branch name"),
    ("docs/superpowers", "an internal document tree"),
    ("handoffs/", "an internal document tree"),
    (".superpowers", "an internal document tree"),
    ("verdict:", "an internal review verdict"),
    ("blocked", "an internal review verdict"),
    ("targeting 1.", "a roadmap commitment for unshipped work"),
    ("targeting v1.", "a roadmap commitment for unshipped work"),
    ("codex", "an internal tooling name"),
)

#: Case identifiers must never reach a public document either.  Same rule the
#: certification path already enforces on docs/public.
#:
#: A forcing product is not a case identifier and is not listed here.  A
#: release note is free to say ERA5, GFS, HRRR or 20CRv3, because each of
#: those names a route a user selects, the same way the CLI does (owner
#: ruling, 2026-08-10).  ``1974`` stays, and still guards the case that the
#: 20CRv3 route was first driven for.
CASE_TOKENS = ("real74", "1974", "ohio", "oklahoma", "may1999")

#: Forcing routes a release note is free to name.  Listed here so the ruling
#: above is enforced rather than only recorded, and re-adding one to
#: CASE_TOKENS fails a test instead of silently blocking a release note from
#: saying which source it is about.
FORCING_ROUTES = ("era5", "gfs", "hrrr", "20cr")


def _newest_section() -> tuple[str, str]:
    """(version heading, body) of the top-most release in the changelog."""
    text = CHANGELOG.read_text(encoding="utf-8")
    starts = [m.start() for m in re.finditer(r"^## ", text, re.MULTILINE)]
    assert len(starts) >= 2, "changelog has fewer than two release sections"
    section = text[starts[0]:starts[1]]
    heading = section.splitlines()[0].strip()
    return heading, section


def test_this_test_reads_a_real_changelog_section() -> None:
    heading, body = _newest_section()
    assert heading.startswith("## "), heading
    assert len(body.split()) > 20, "newest section is suspiciously empty"


def test_the_newest_release_note_is_not_a_development_log() -> None:
    heading, body = _newest_section()
    words = len(body.split())
    limit = word_limit(heading)
    assert words <= limit, (
        f"{heading}: {words} words of release notes (limit {limit}). "
        "A release note says what shipped; put the reasoning in the repository "
        "and link to it.")


def test_only_a_named_release_gets_more_room() -> None:
    assert word_limit("## 2.8.0 (2026-09-29)") == 7000
    assert word_limit("## 2.8.1 (2026-10-01)") == MAX_WORDS
    assert word_limit("## 2.7.6 (2026-09-22)") == MAX_WORDS
    assert word_limit("## 12.8.0 (2030-01-01)") == MAX_WORDS
    assert word_limit("## 2.8.0-rc1") == MAX_WORDS


@pytest.mark.parametrize("needle,what", LEAKS)
def test_the_newest_release_note_leaks_no_internal_state(needle: str,
                                                         what: str) -> None:
    heading, body = _newest_section()
    lowered = body.lower()
    if needle not in lowered:
        return
    line = next(l.strip() for l in body.splitlines()
                if needle in l.lower())
    pytest.fail(
        f"{heading} contains {needle!r} -- {what} -- in: {line[:160]!r}")


@pytest.mark.parametrize("route", FORCING_ROUTES)
def test_a_forcing_route_is_not_a_case_identifier(route: str) -> None:
    assert route not in CASE_TOKENS, (
        f"{route!r} names a forcing route a user selects on the command line, "
        "so a release note is allowed to say it. This list guards case "
        "identifiers, which name one experiment nobody outside the project "
        "runs.")


@pytest.mark.parametrize("token", CASE_TOKENS)
def test_the_newest_release_note_names_no_case_identifier(token: str) -> None:
    heading, body = _newest_section()
    assert token not in body.lower(), (
        f"{heading} names the case identifier {token!r}; case names never "
        "reach a public document.")
