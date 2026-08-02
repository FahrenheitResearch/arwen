"""Verify that a tagged source tree does not impersonate release artifacts.

The release workflow mutates a private build workspace with pins computed
from target-native bundle bytes before building the supported wheel and
sdist. GitHub's automatic source archives, however, are snapshots of the
tagged tree. Their packaged pins document must therefore stay explicitly
unpinned instead of pointing at a candidate or a different release.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

PINS_SCHEMA = "gpuwm-bridge-pins-v1"
REQUIRED_KEYS = frozenset({"schema", "release", "platforms", "note"})
DEFAULT_PINS = (
    Path(__file__).resolve().parents[1]
    / "gpuwm" / "data" / "bridges" / "bridge-pins.json"
)


class SourceBridgePinsError(ValueError):
    """The source-tree pins document could misrepresent release bytes."""


def verify_source_pins(path: Path = DEFAULT_PINS) -> dict[str, Any]:
    """Return a valid, explicitly-unpinned source document or refuse it."""

    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SourceBridgePinsError(
            f"source bridge pins are unreadable at {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise SourceBridgePinsError("source bridge pins must be a JSON object")
    keys = frozenset(payload)
    if keys != REQUIRED_KEYS:
        raise SourceBridgePinsError(
            "source bridge pins keys drifted "
            f"(missing {sorted(REQUIRED_KEYS - keys)}, "
            f"extra {sorted(keys - REQUIRED_KEYS)})"
        )
    if payload["schema"] != PINS_SCHEMA:
        raise SourceBridgePinsError(
            f"source bridge pins schema must be {PINS_SCHEMA!r}"
        )
    if payload["release"] is not None:
        raise SourceBridgePinsError(
            "tagged source tree must have release=null; release pins are "
            "generated only in the distribution build workspace"
        )
    if payload["platforms"] != {}:
        raise SourceBridgePinsError(
            "tagged source tree must have platforms={}; bundle pins are "
            "generated only from target-native release bytes"
        )
    if not isinstance(payload["note"], str) or not payload["note"].strip():
        raise SourceBridgePinsError("source bridge pins note must be non-empty")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_PINS)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        verify_source_pins(args.path)
    except SourceBridgePinsError as error:
        parser.error(str(error))
    print(f"source bridge pins: PASS ({args.path})")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
