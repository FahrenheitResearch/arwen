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
SINCE..UNTIL``, merges included).  ``--write`` adds those documents while
preserving the retained history required by old preparation receipts.
Without it the selected record is printed.  ``--check``
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
            # git am preserves this blob but changes an exported lane's
            # commit ID when its parent is the full engine repository.
            "registry_blob_sha1": _git("rev-parse", f"{commit}:{REGISTRY}").strip(),
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


def retain_history(previous: dict, current: dict) -> dict:
    """A narrow intake range must not discard older preparation identities."""
    for key in ("schema", "identity_schema"):
        if previous.get(key) != current[key]:
            raise ValueError(f"registry history {key} differs; refusing to discard old identities")
    for digest in previous["physics"].keys() & current["physics"].keys():
        if previous["physics"][digest] != current["physics"][digest]:
            raise ValueError(f"registry physics {digest} has conflicting parts; refusing false identity")
    for digest in previous["documents"].keys() & current["documents"].keys():
        if previous["documents"][digest]["physics_sha256"] != current["documents"][digest]["physics_sha256"]:
            raise ValueError(f"registry document {digest} has conflicting physics identities")
    return {**current,
            "documents": {**previous["documents"], **current["documents"]},
            "physics": {**previous["physics"], **current["physics"]}}


def check() -> list[str]:
    saved = json.loads(REGISTRY_PHYSICS_HISTORY_PATH.read_text(
        encoding="utf-8"))
    failures: list[str] = []
    for digest, row in sorted(saved["documents"].items()):
        commit = row["commit"]
        try:
            document = _document(commit)
        except subprocess.CalledProcessError:
            blob = row.get("registry_blob_sha1")
            if not isinstance(blob, str) or len(blob) != 40 or any(
                    char not in "0123456789abcdef" for char in blob):
                failures.append(f"{digest}: commit {commit} is not in this "
                                "repository's history and no registry blob is recorded")
                continue
            try:
                document = json.loads(_git("cat-file", "blob", blob))
            except (subprocess.CalledProcessError, json.JSONDecodeError):
                failures.append(f"{digest}: recorded registry blob {blob} "
                                "is missing or is not a registry document")
                continue
            if not isinstance(document, dict):
                failures.append(f"{digest}: recorded registry blob {blob} "
                                "is not a registry object")
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
    saved = record(_commits_between(args.since, args.until))
    if args.write and REGISTRY_PHYSICS_HISTORY_PATH.exists():
        saved = retain_history(json.loads(REGISTRY_PHYSICS_HISTORY_PATH.read_text(
            encoding="utf-8")), saved)
    text = canonical_json(saved)
    if args.write:
        # Bytes, so Windows writes the same LF-only file Linux does.
        REGISTRY_PHYSICS_HISTORY_PATH.write_bytes(
            (text + "\n").encode("utf-8"))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
