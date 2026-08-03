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
MAX_WORDS = 3000

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
CASE_TOKENS = ("real74", "1974", "ohio", "oklahoma", "may1999", "20cr")


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
    assert words <= MAX_WORDS, (
        f"{heading}: {words} words of release notes (limit {MAX_WORDS}). "
        "A release note says what shipped; put the reasoning in the repository "
        "and link to it.")


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


@pytest.mark.parametrize("token", CASE_TOKENS)
def test_the_newest_release_note_names_no_case_identifier(token: str) -> None:
    heading, body = _newest_section()
    assert token not in body.lower(), (
        f"{heading} names the case identifier {token!r}; case names never "
        "reach a public document.")
