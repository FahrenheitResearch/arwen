"""Re-resolve every ``file:line`` anchor the manual and the public pages cite.

THE CONCRETE BREAKAGE THIS PREVENTS.  A documentation sentence that cites a
line number is making a checkable claim, and line numbers rot the moment
anything above them is edited.  Measured on this tree, 2026-09-17:
``docs/manual/02-dynamics-grids-nesting.md`` cited
``docs/public/CONFIGURATION.md:235`` and ``:414`` for the ``scalar_adv_opt``
matching rule; :235 is the middle of a plain list of other key names and
:414 is a sentence about sealed preparation states, while the rows that
carry the rule are :332 and :627.  Two lines lower it cited :90 for
``spec_bdy_width``, which is the ``## Tweakable knobs`` heading; the row is
:103.  A reader who follows such a citation to check a claim finds
unrelated text and cannot tell whether the claim or the pointer is wrong.

WHY AN ANCHOR AND NOT EXISTENCE.  :235 exists.  So each citation is resolved
against an ANCHOR: the configuration key or code symbol the citing sentence
itself names, in backticks.  The cited line, or any line of a cited range,
must contain one of those names.  A citation that drifts by one line loses
its anchor and is reported here.  This is the rule
``tools/check_registry_citations.py`` applies to the physics registry's own
citations, with the anchor read out of the citing prose instead of recorded
in a table, because prose is where these live and a table would rot beside
the thing it describes.

SCOPE.  ``docs/manual/`` and ``docs/public/``, citing ``CONFIGURATION.md`` or
any ``gpuwm/**.py`` module: the pages a reader is sent to, and the two kinds
of target whose lines a reader would actually open.  A citation into a file
this repository does not carry is reported as unresolvable rather than
guessed at.  A sentence that names no key or symbol is counted as unanchored
and listed without failing, because nothing can check such a citation and
saying how many there are is the useful answer.

    python tools/check_doc_anchors.py            # report, exit 1 on offences
    python tools/check_doc_anchors.py --quiet    # counts only
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The documentation trees whose citations are checked.
DOC_ROOTS = ("docs/manual", "docs/public")

#: ``path:12``, ``path:12-14``, ``path:12, 14``, ``path:12 and :14``.  The
#: continuation forms matter: the tree writes ``:235, 414`` and
#: ``:3872-3873 ... and :4618``, and reading only the first number would
#: check a third of the citations while reporting the rest as checked.
_CITATION = re.compile(
    r"(?P<path>(?:docs|gpuwm|tools|tilestream)/[A-Za-z0-9_./-]+?"
    r"\.(?:md|py|cu|rs|toml))"
    r":(?P<first>\d+(?:-\d+)?)"
    r"(?P<rest>(?:\s*(?:,|and)\s*:?\d+(?:-\d+)?)*)")

#: ``code`` spans in the citing prose.  A span is taken as an anchor only if
#: it reads as a key or a symbol rather than a phrase, so a sentence that
#: backticks a whole namelist line does not anchor on a space.
_CODE_SPAN = re.compile(r"`([^`\n]{2,80})`")

#: An anchor candidate is IDENTIFIER-SHAPED, not merely backticked.  This
#: bar is the difference between an instrument and a noise generator:
#: without it the survey read ``and``, ``then``, ``flag``, ``required``
#: and ``rust`` out of ordinary prose as if they were symbols, and called
#: 41 of 48 citations drifted on anchors their sentences never meant.  A
#: configuration key or code symbol in this tree carries an underscore, a
#: capital or a digit, and is at least four characters.
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{3,}$")
_SPECIFIC = re.compile(r"[_A-Z0-9]")

#: Sentence end, for the span a citation's anchors are read from: the
#: citation's own sentence, widened to its paragraph when that sentence
#: backticks nothing.
_SENTENCE_END = re.compile(r"(?<=[.:;])\s")


def is_in_scope(path: str) -> bool:
    """Whether this checker resolves a citation into ``path``."""

    if path == "docs/public/CONFIGURATION.md":
        return True
    return path.startswith("gpuwm/") and path.endswith(".py")


def _line_numbers(match: "re.Match[str]") -> list[int]:
    numbers: list[int] = []
    pieces = [match.group("first")]
    pieces += re.findall(r"\d+(?:-\d+)?", match.group("rest") or "")
    for piece in pieces:
        if "-" in piece:
            low, high = piece.split("-", 1)
            numbers.extend(range(int(low), int(high) + 1))
        else:
            numbers.append(int(piece))
    return numbers


def _paragraph_bounds(text: str, index: int) -> tuple[int, int]:
    blank = "\n\n"
    start = text.rfind(blank, 0, index)
    start = 0 if start < 0 else start + len(blank)
    end = text.find(blank, index)
    end = len(text) if end < 0 else end
    return start, end


def _names_in(scope: str) -> list[str]:
    names: list[str] = []
    for span in _CODE_SPAN.findall(scope):
        stripped = span.strip()
        if not stripped:
            continue
        for candidate in (stripped, stripped.split()[-1]):
            if (_NAME.match(candidate) and _SPECIFIC.search(candidate)
                    and candidate not in names):
                names.append(candidate)
    return names


#: A phrase the citing sentence QUOTES out of the cited lines is the
#: strongest anchor there is, and it outranks a symbol name.  Measured:
#: the observation battery's B4 receipt cites a module at :43-44 and
#: quotes the comment that sits there, in a sentence that also names an
#: exception class defined in a different file -- so a symbol-only rule
#: called a correct citation drifted.  Whitespace is normalised on both
#: sides because the quote is line-wrapped in the page and is not in the
#: source.
#: An OPENING quotation mark, which is one not preceded by a word
#: character or an equals sign.  Without that bar the pairing walks off:
#: the same paragraph writes ``stock_wrf_export="optional"`` inline, and a
#: plain pair-by-pair scan took that value's closing mark as an opening
#: one, swallowed the sentence after it and never saw the quoted comment
#: that the citation is anchored on.  Newlines are admitted inside the
#: phrase because the page wraps its quotes and the source does not.
_QUOTE = re.compile(r"(?<![A-Za-z0-9_=])\"([^\"]{20,200})\"")


def _squeeze(text: str) -> str:
    return " ".join(text.split())


def anchor_quotes(text: str, match: "re.Match[str]") -> list[str]:
    """Phrases the citing PARAGRAPH quotes out of its cited lines.

    The paragraph rather than the sentence, and the phrase with its
    trailing punctuation removed.  Both are what the tree actually writes:
    the quote arrives after a colon, which ends the sentence span by this
    module's own reckoning, and a page closes the quotation with a full
    stop the source line does not carry.  A phrase of twenty characters or
    more that appears verbatim at the cited lines is not a coincidence,
    whichever sentence of the paragraph introduced it.
    """

    start, end = _paragraph_bounds(text, match.start())
    found = []
    for quote in _QUOTE.findall(text[start:end]):
        squeezed = _squeeze(quote).rstrip(".,;:")
        if len(squeezed) >= 20:
            found.append(squeezed)
    return found


def anchor_names(text: str, match: "re.Match[str]") -> list[str]:
    """The key or symbol names the citing sentence claims something about."""

    start, end = _paragraph_bounds(text, match.start())
    paragraph = text[start:end]
    offset = match.start() - start
    sentence_start = 0
    for boundary in _SENTENCE_END.finditer(paragraph[:offset]):
        sentence_start = boundary.end()
    sentence = paragraph[sentence_start:offset + (match.end() - match.start())]
    # THE CITING SENTENCE ONLY, not its paragraph.  A paragraph-wide read
    # was tried and is wrong: it attributed one sentence's symbols to
    # another sentence's citation and reported drift on pointers that were
    # never about those names.  A citation whose own sentence names nothing
    # is unanchored, which is a statement this checker is willing to make.
    return _names_in(sentence)


#: A receipt says which revision it was measured against, in its opening
#: lines: "Written 2026-08-03 against `fc15d9ae`".  Such a document's
#: citations are EVIDENCE AT THAT REVISION, so they are resolved there.
#: Resolving them against HEAD reports drift that is not drift, and the
#: only way to make that report green would be to re-point a receipt at
#: lines its run never read -- which falsifies the receipt instead of
#: fixing anything.  Measured: four rows of the observation battery's
#: receipts, all of them correct at the revision each one names.
_PINNED = re.compile(r"against\s+`([0-9a-f]{7,40})`")


def revision_of(text: str) -> str | None:
    """The revision a document pins its citations to, when it states one."""

    head = "\n".join(text.splitlines()[:12])
    found = _PINNED.search(head)
    return found.group(1) if found is not None else None


def _target_lines(path: str, revision: str | None = None) -> list[str] | None:
    if revision is not None:
        import subprocess
        try:
            raw = subprocess.check_output(
                ["git", "show", revision + ":" + path],
                cwd=REPO_ROOT, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError):
            return None
        return raw.decode("utf-8", "replace").splitlines()
    target = REPO_ROOT / path
    if not target.is_file():
        return None
    body = target.read_text(encoding="utf-8", errors="replace")
    return body.splitlines()


def _line_of(text: str):
    starts = [0]
    for index, character in enumerate(text):
        if character == "\n":
            starts.append(index + 1)

    def resolve(index: int) -> int:
        low, high = 0, len(starts) - 1
        while low < high:
            middle = (low + high + 1) // 2
            if starts[middle] <= index:
                low = middle
            else:
                high = middle - 1
        return low + 1

    return resolve


def survey(root: Path = REPO_ROOT) -> dict:
    """Resolve every in-scope citation under the documentation roots."""

    drifted: list[str] = []
    unresolvable: list[str] = []
    unanchored: list[str] = []
    pinned = 0
    checked = 0

    for doc_root in DOC_ROOTS:
        for doc in sorted((root / doc_root).rglob("*.md")):
            relative = doc.relative_to(root).as_posix()
            text = doc.read_text(encoding="utf-8")
            revision = revision_of(text)
            line_of = _line_of(text)
            for match in _CITATION.finditer(text):
                path = match.group("path")
                if not is_in_scope(path):
                    continue
                where = relative + ":" + str(line_of(match.start()))
                lines = _target_lines(path, revision)
                if lines is None and revision is not None:
                    unresolvable.append(
                        where + " cites " + path + " at the revision "
                        + revision + " this document pins itself to, and that "
                        "revision is not in this clone")
                    continue
                if lines is None:
                    unresolvable.append(
                        where + " cites " + path + ", which is not in this tree")
                    continue
                numbers_for_quote = _line_numbers(match)
                quoted_text = _squeeze("\n".join(
                    lines[number - 1] for number in numbers_for_quote
                    if 1 <= number <= len(lines)))
                quotes = [quote for quote in anchor_quotes(text, match)
                          if quote in quoted_text]
                if quotes:
                    checked += 1
                    if revision is not None:
                        pinned += 1
                    continue
                names = anchor_names(text, match)
                if not names:
                    unanchored.append(
                        where + " cites " + path + " and names no key or symbol")
                    continue
                # A name the target file does not carry ANYWHERE is not a
                # drifted anchor: the sentence names something belonging to
                # a different file, and calling that drift would send a
                # reader hunting for a line that cannot exist.
                body = "\n".join(lines)
                names = [name for name in names if name in body]
                if not names:
                    unanchored.append(
                        where + " cites " + path
                        + " and names no symbol that file carries")
                    continue
                numbers = _line_numbers(match)
                cited = "\n".join(lines[number - 1] for number in numbers
                                  if 1 <= number <= len(lines))
                checked += 1
                if revision is not None:
                    pinned += 1
                if any(name in cited for name in names):
                    continue
                found = {name: [index + 1 for index, line in enumerate(lines)
                                if name in line][:3] for name in names}
                drifted.append(
                    where + ": cites " + path + ":"
                    + ",".join(str(number) for number in numbers)
                    + " for " + repr(names)
                    + ", and no cited line carries any of them; they are at "
                    + repr(found))

    return {"checked": checked, "pinned": pinned, "drifted": drifted,
            "unresolvable": unresolvable, "unanchored": unanchored}


def check(quiet: bool = False) -> int:
    """Print the survey.  Returns the number of offences."""

    result = survey()
    if not quiet:
        for row in result["drifted"]:
            print("DRIFTED     " + row)
        for row in result["unresolvable"]:
            print("UNRESOLVED  " + row)
        for row in result["unanchored"]:
            print("UNANCHORED  " + row)
    print("checked {checked} anchored citations ({pinned} against the "
          "revision their document pins): {drifted} drifted, "
          "{unresolvable} unresolvable, {unanchored} unanchored".format(
              checked=result["checked"],
              pinned=result["pinned"],
              drifted=len(result["drifted"]),
              unresolvable=len(result["unresolvable"]),
              unanchored=len(result["unanchored"])))
    return len(result["drifted"]) + len(result["unresolvable"])


def main() -> int:
    parser = argparse.ArgumentParser(description="check documentation anchors")
    parser.add_argument("--quiet", action="store_true",
                        help="print the counts without the rows")
    arguments = parser.parse_args()
    return 1 if check(quiet=arguments.quiet) else 0


if __name__ == "__main__":
    sys.exit(main())
