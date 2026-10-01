"""Write or check gpuwm/physics_registry_history.json (A153).

A selection receipt written before receipts carried ``registry_physics``
names only the registry DOCUMENT digest it was prepared under.  The
history record maps such a document digest to that document's physics
parts (``gpuwm.physics_registry.registry_physics_parts``), and this tool
computes it from the documents git holds, so every row is a fact about a
committed document rather than a value typed by hand.

Usage
-----
    python tools/registry_physics_history.py --since REV --until REV --write
    python tools/registry_physics_history.py --check

``--since/--until`` takes the document at ``--since`` and every document
the registry took after it up to ``--until`` (``git rev-list
SINCE..UNTIL``, merges included).  ``--write`` rewrites the record with
exactly those documents; without it the record is printed.  ``--check``
recomputes every row from a committed document with that digest and
fails on a row it cannot find or that disagrees; it needs the history.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

MODEL = pathlib.Path(__file__).resolve().parents[1]
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

from gpuwm.physics_registry import (  # noqa: E402
    REGISTRY_PHYSICS_HISTORY_PATH,
    REGISTRY_PHYSICS_HISTORY_SCHEMA,
    REGISTRY_PHYSICS_IDENTITY_SCHEMA,
    canonical_json,
    registry_physics_parts,
    registry_physics_sha256,
    registry_sha256,
)

REGISTRY = "gpuwm/physics_registry_v2.json"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=MODEL, check=True, capture_output=True,
        text=True, encoding="utf-8").stdout


def _document(commit: str) -> dict:
    return json.loads(_git("show", f"{commit}:{REGISTRY}"))


def record(commits: list[str]) -> dict[str, object]:
    """The history record for the registry documents at ``commits``."""

    documents: dict[str, dict[str, str]] = {}
    physics: dict[str, dict[str, str]] = {}
    for commit in commits:
        document = _document(commit)
        digest = registry_sha256(document)
        if digest in documents:
            continue
        physics_digest = registry_physics_sha256(document)
        documents[digest] = {
            "physics_sha256": physics_digest,
            "commit": _git("rev-parse", "--short=10", commit).strip(),
        }
        physics.setdefault(physics_digest, registry_physics_parts(document))
    return {
        "schema": REGISTRY_PHYSICS_HISTORY_SCHEMA,
        "identity_schema": REGISTRY_PHYSICS_IDENTITY_SCHEMA,
        "documents": documents,
        "physics": physics,
    }


def _commits_between(since: str, until: str) -> list[str]:
    return [since, *_git("rev-list", "--reverse", f"{since}..{until}",
                         "--", REGISTRY).split()]


def check() -> list[str]:
    saved = json.loads(REGISTRY_PHYSICS_HISTORY_PATH.read_text(
        encoding="utf-8"))
    failures: list[str] = []
    for digest, row in sorted(saved["documents"].items()):
        commit = row["commit"]
        try:
            document = _document(commit)
        except subprocess.CalledProcessError:
            failures.append(f"{digest}: commit {commit} is not in this "
                            "repository's history")
            continue
        if registry_sha256(document) != digest:
            failures.append(f"{digest}: the document at {commit} has "
                            f"digest {registry_sha256(document)}")
            continue
        physics = registry_physics_sha256(document)
        if physics != row["physics_sha256"]:
            failures.append(f"{digest}: physics {physics}, the record says "
                            f"{row['physics_sha256']}")
        elif saved["physics"].get(physics) != registry_physics_parts(
                document):
            failures.append(f"{digest}: the recorded parts of {physics} "
                            "differ from the document's")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--since")
    parser.add_argument("--until")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        failures = check()
        for line in failures:
            print(line)
        print(f"{len(failures)} failing")
        return 1 if failures else 0
    if not (args.since and args.until):
        parser.error("--since and --until are required without --check")
    text = canonical_json(record(_commits_between(args.since, args.until)))
    if args.write:
        # Bytes, so Windows writes the same LF-only file Linux does.
        REGISTRY_PHYSICS_HISTORY_PATH.write_bytes(
            (text + "\n").encode("utf-8"))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
