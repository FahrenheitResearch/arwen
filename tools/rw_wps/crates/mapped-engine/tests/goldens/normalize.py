"""Rewrite the golden JSON files with LF line endings.

`extract.py` writes through Python's text layer, which on Windows turns
every newline into CRLF.  The repository stores text as LF (`.gitattributes`
`* -text`, and `tests/test_line_ending_stability.py` holds the promise), so
a golden written on Windows would land as a CRLF file in a tree that
promises otherwise.  `extract.py` calls this at the end; it is also
runnable on its own after an edit.
"""

from __future__ import annotations

import pathlib


def normalize(path: pathlib.Path) -> bool:
    """Rewrite `path` with LF endings.  True when bytes changed."""

    data = path.read_bytes()
    converted = data.replace(b"\r\n", b"\n")
    if converted == data:
        return False
    path.write_bytes(converted)
    return True


def main() -> int:
    here = pathlib.Path(__file__).resolve().parent
    for path in sorted(here.glob("*.json")):
        if normalize(path):
            print(f"normalized {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
