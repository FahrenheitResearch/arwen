"""Every published sentence that names resolved tornado structure negates it.

The shipped docs describe a 500 m innermost grid and a severe-weather
diagnostic suite built on it.  Those are supercell-environment and
mesocyclone-proxy statements.  The failure mode this guard exists for is a
later edit -- a re-wrap, a summary line, a marketing pass -- that drops the
negation and leaves an affirmative resolved-tornado claim standing in
``README.md`` or under ``docs/public/``.

The invariant is enforced at two scopes, and both must hold for every
occurrence of :data:`RESOLVED_PATTERN`:

*sentence scope*
    The Markdown block is normalized (lines joined, whitespace runs
    collapsed) and split into sentences; a negation token must appear
    earlier in the sentence that carries the occurrence.

*line scope*
    A negation token must also appear earlier on the physical source line
    that carries the occurrence.

Sentence scope alone is not discriminating enough to be a gate: a sentence
holding two occurrences and one negation keeps passing when the second
negation is deleted, because the first one still sits earlier in the
sentence.  Line scope catches exactly that, which is why the disclaimer
prose is hard-wrapped so each occurrence carries its own negation on its own
line.  Neither scope replaces the other and neither is optional.

Run it directly for a report::

    python tools/check_negation_invariant.py --json report.json

Exit status is 0 when every occurrence is negated at both scopes, 1
otherwise.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

#: Occurrences that must be negated.  Kept identical to the pattern the
#: acceptance criterion states, matched case-insensitively.
RESOLVED_PATTERN = re.compile(
    r"tornado-resolv|resolv[a-z]* tornado|resolved-tornado",
    re.IGNORECASE,
)

#: Negation tokens ("not", "NOT", "No", "no "), bounded so that "cannot",
#: "note" and "another" do not count as negations.
NEGATION_PATTERN = re.compile(r"(?<![A-Za-z])(not|NOT|No|no)(?![A-Za-z])")

#: Sentence terminator: a period/question/exclamation mark followed by
#: whitespace or end of block.
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

LIST_ITEM = re.compile(r"^\s{0,3}(?:[-*+]|\d+[.)])\s+")

_FENCE = re.compile(r"^\s*(```|~~~)")


@dataclass(frozen=True)
class Occurrence:
    """One pattern hit and the verdict at each enforcement scope."""

    path: str
    line: int
    match: str
    line_text: str
    sentence: str
    negated_in_line: bool
    negated_in_sentence: bool

    @property
    def ok(self) -> bool:
        return self.negated_in_line and self.negated_in_sentence


@dataclass(frozen=True)
class Block:
    """A Markdown list item or paragraph, with its source line numbers."""

    lines: tuple[str, ...]
    first_line: int

    def normalized(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self.lines)).strip()


def iter_blocks(text: str) -> Iterator[Block]:
    """Split Markdown into list items and paragraphs.

    A list item runs from its bullet line through every following line
    until the next bullet or a blank line.  A paragraph is any other run of
    contiguous non-blank lines.  Fenced code is skipped: it is machine
    input, not a claim.
    """
    lines = text.splitlines()
    buf: list[str] = []
    start = 0
    in_fence = False

    def flush() -> Iterator[Block]:
        if buf:
            yield Block(tuple(buf), start)

    index = 0
    while index < len(lines):
        line = lines[index]
        if _FENCE.match(line):
            yield from flush()
            buf = []
            in_fence = not in_fence
            index += 1
            continue
        if in_fence:
            index += 1
            continue
        if not line.strip():
            yield from flush()
            buf = []
            index += 1
            continue
        if LIST_ITEM.match(line) and buf:
            yield from flush()
            buf = []
        if not buf:
            start = index + 1
        buf.append(line)
        index += 1
    yield from flush()


def sentence_spans(normalized: str) -> list[tuple[int, int]]:
    """Half-open (start, end) spans of every sentence in a normalized block."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for sentence in SENTENCE_SPLIT.split(normalized):
        spans.append((cursor, cursor + len(sentence)))
        # +1 for the single space the split consumed.
        cursor += len(sentence) + 1
    return spans


def _sentence_of(
    normalized: str, spans: Sequence[tuple[int, int]], position: int
) -> tuple[int, int]:
    for start, end in spans:
        if start <= position < end:
            return start, end
    return 0, len(normalized)


def scan_text(text: str, path: str) -> list[Occurrence]:
    """Report every :data:`RESOLVED_PATTERN` hit in one document.

    Normalization preserves character order, so the *k*-th hit found while
    walking physical lines is the *k*-th hit in the normalized block.  That
    pairing is what lets one occurrence carry both a line verdict and a
    sentence verdict without a fragile offset map.
    """
    found: list[Occurrence] = []
    for block in iter_blocks(text):
        normalized = block.normalized()
        normalized_hits = list(RESOLVED_PATTERN.finditer(normalized))
        if not normalized_hits:
            continue
        spans = sentence_spans(normalized)
        line_hits = [
            (offset, raw, hit)
            for offset, raw in enumerate(block.lines)
            for hit in RESOLVED_PATTERN.finditer(raw)
        ]
        for index, (offset, raw, hit) in enumerate(line_hits):
            if index < len(normalized_hits):
                norm_hit = normalized_hits[index]
                start, end = _sentence_of(normalized, spans, norm_hit.start())
                sentence = normalized[start:end]
                before_in_sentence = normalized[start:norm_hit.start()]
            else:  # pragma: no cover - normalization cannot lose a hit
                sentence = normalized
                before_in_sentence = ""
            found.append(
                Occurrence(
                    path=path,
                    line=block.first_line + offset,
                    match=hit.group(0),
                    line_text=raw.strip(),
                    sentence=sentence,
                    negated_in_line=bool(
                        NEGATION_PATTERN.search(raw[: hit.start()])
                    ),
                    negated_in_sentence=bool(
                        NEGATION_PATTERN.search(before_in_sentence)
                    ),
                )
            )
    return found


def documents(root: Path) -> list[Path]:
    """README plus every published Markdown page, in a stable order."""
    paths: list[Path] = []
    readme = root / "README.md"
    if readme.is_file():
        paths.append(readme)
    paths.extend(sorted((root / "docs" / "public").glob("*.md")))
    return paths


def check(root: Path, paths: Sequence[Path] | None = None) -> dict:
    """Build the machine report for one tree."""
    targets = list(paths) if paths is not None else documents(root)
    occurrences: list[Occurrence] = []
    for path in targets:
        occurrences.extend(
            scan_text(
                path.read_text(encoding="utf-8"),
                path.relative_to(root).as_posix(),
            )
        )
    failures = [item for item in occurrences if not item.ok]
    return {
        "tool": "check_negation_invariant",
        "pattern": RESOLVED_PATTERN.pattern,
        "negation_tokens": ["not", "NOT", "No", "no "],
        "scopes": ["line", "sentence"],
        "documents_scanned": [
            path.relative_to(root).as_posix() for path in targets
        ],
        "occurrences": [asdict(item) for item in occurrences],
        "occurrence_count": len(occurrences),
        "failure_count": len(failures),
        "verdict": "PASS" if not failures else "FAIL",
    }


def _render(report: dict, stream) -> None:
    print(
        f"{report['verdict']}: {report['occurrence_count']} occurrence(s), "
        f"{report['failure_count']} unnegated",
        file=stream,
    )
    for item in report["occurrences"]:
        flag = "ok  " if item["negated_in_line"] and item[
            "negated_in_sentence"
        ] else "FAIL"
        print(
            f"  {flag} {item['path']}:{item['line']} "
            f"[{item['match']}] line={item['negated_in_line']} "
            f"sentence={item['negated_in_sentence']}",
            file=stream,
        )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="tree to scan (default: this checkout)",
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="write the machine report here"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = check(args.root.resolve())
    if args.json is not None:
        args.json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    _render(report, sys.stdout)
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
