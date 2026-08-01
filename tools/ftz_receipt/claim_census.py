"""Register every FTZ / subnormal claim in the public tree.

Scope is machine-defined: ``git ls-files`` minus the RELEASE-EXCLUDE globs
minus the two vendored Rust trees, tests INCLUDED.  A hand-kept list would
drift, and a sentence that drifts out of the census is a public claim nobody
is checking.

Each record pins one line by its exact text, so editing a registered sentence
without re-running this tool fails the consistency gate.  ``asserted_token``
binds a record to the receipt's verdict for its (route, mechanism) cell.
Attribution across the five routes is judgment, not mechanics: this tool
never guesses one.  A record it cannot attribute is written with
``asserted_token: null`` and ``attribution: "unattributed"``, and the gate
requires only that a token, once present, agrees with the receipt.

Usage::

    python -m tools.ftz_receipt.claim_census
    python -m tools.ftz_receipt.claim_census --check
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from tools.ftz_receipt import route_inventory as ri

SCHEMA_ID = "gpuwm.ftz-claim-census/v1"

CLAIM_PATTERN = re.compile(r"ftz|subnormal", re.IGNORECASE)

#: Vendored third-party trees.  Their arithmetic is not ours to describe.
VENDOR_PREFIXES = ("tools/grib1_bridge/vendor/", "tools/rustwx/vendor/")

#: The receipt itself.  Registering a measurement against itself is circular:
#: the bit table and the receipt are the authority these records point AT, so
#: they are not among the claims the census checks.  Everything else stays in
#: scope, the probe and this tool's own tests included.
SELF_PREFIXES = ("tools/ftz_receipt/receipt/", "tools/ftz_receipt/fixtures/")

#: Files whose FTZ mention motivates shipped behaviour rather than describing
#: it.  They are REGISTERED here and left untouched: revisiting any of them
#: changes numbers and breaks bitwise gates, which is [D-36], not this work.
BEHAVIOURAL_FILES = (
    "gpuwm/core/kernels/rrtmg_sw.cu",
    "gpuwm/core/rrtmg_legacy_prep.py",
    "tools/grib1_bridge/src/lib.rs",
    "gpuwm/ingest/hrrr.py",
)

ATTRIBUTION_PENDING = "unattributed"
ATTRIBUTION_SET = "attributed"

TEXT_SUFFIXES = frozenset({
    ".py", ".pyi", ".md", ".txt", ".rst", ".toml", ".cfg", ".ini", ".json",
    ".jsonl", ".csv", ".cu", ".cuh", ".c", ".h", ".cpp", ".rs", ".F", ".F90",
    ".f90", ".sh", ".bat", ".ps1", ".yml", ".yaml", ".cff",
})


def release_exclusions(root: Path) -> list[str]:
    path = root / "RELEASE-EXCLUDE.txt"
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8")
            .splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def is_excluded(relpath: str, globs: list[str]) -> bool:
    if relpath.startswith(VENDOR_PREFIXES) or relpath.startswith(
            SELF_PREFIXES):
        return True
    for pattern in globs:
        if fnmatch.fnmatch(relpath, pattern):
            return True
        if pattern.endswith("/**") and relpath.startswith(pattern[:-2]):
            return True
    return False


def public_files(root: Path) -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=str(root),
                         capture_output=True, text=True, check=True)
    globs = release_exclusions(root)
    return sorted(path for path in out.stdout.splitlines()
                  if path.strip() and not is_excluded(path, globs))


def anchor_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def collect_sites(root: Path, files: list[str],
                  census_path: str | None = None) -> list[dict]:
    """One record per matching line, in path then line order."""
    records: list[dict] = []
    for relpath in files:
        if relpath == census_path:
            continue
        path = root / relpath
        if path.suffix and path.suffix not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if not CLAIM_PATTERN.search(line):
                continue
            anchor = line.strip()
            records.append({
                "file": relpath,
                "line": number,
                "kind": ("behavioural" if relpath in BEHAVIOURAL_FILES
                         else "prose"),
                "anchor": anchor,
                "anchor_sha256": anchor_sha256(anchor),
                "route": None,
                "mechanism": None,
                "asserted_token": None,
                "attribution": ATTRIBUTION_PENDING,
            })
    return records


def merge_attribution(records: list[dict],
                      previous: dict | None) -> list[dict]:
    """Carry a curator's attribution forward; never invent one."""
    if not previous:
        return records
    index = {(item["file"], item["anchor_sha256"]): item
             for item in previous.get("sites", [])}
    for record in records:
        prior = index.get((record["file"], record["anchor_sha256"]))
        if prior is None:
            continue
        for key in ("route", "mechanism", "asserted_token", "attribution"):
            if prior.get(key) is not None:
                record[key] = prior[key]
    return records


def build_census(root: Path, previous: dict | None = None) -> dict:
    census_rel = "gpuwm/verify/ftz_claim_sites.json"
    records = merge_attribution(
        collect_sites(root, public_files(root), census_rel), previous)
    by_kind: dict[str, int] = {}
    for record in records:
        by_kind[record["kind"]] = by_kind.get(record["kind"], 0) + 1
    return {
        "schema": SCHEMA_ID,
        "generator": "tools/ftz_receipt/claim_census.py",
        "scope": "git ls-files minus RELEASE-EXCLUDE.txt globs minus "
                 "vendored trees; tests included",
        "pattern": CLAIM_PATTERN.pattern,
        "behavioural_files": list(BEHAVIOURAL_FILES),
        "attribution_note":
            "route/mechanism attribution is judgment across five measured "
            "routes and is not machine-derivable; unattributed records carry "
            "asserted_token: null and the gate requires only that a token, "
            "once written, agrees with the receipt",
        "site_count": len(records),
        "site_count_by_kind": dict(sorted(by_kind.items())),
        "sites": records,
    }


def render(document: dict) -> str:
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def default_output(root: Path) -> Path:
    return root / "gpuwm" / "verify" / "ftz_claim_sites.json"


def check_census(root: Path, census: dict,
                 receipt: dict | None = None) -> list[str]:
    """Return every reason the census does not describe the tree.

    Three failure classes, matching the three mutations the test drives: an
    unregistered claim line, an anchor whose text no longer matches, and an
    asserted token the receipt does not support.
    """
    problems: list[str] = []
    census_rel = "gpuwm/verify/ftz_claim_sites.json"
    found = {(record["file"], record["line"]): record
             for record in collect_sites(root, public_files(root),
                                         census_rel)}
    registered = {(record["file"], record["line"]): record
                  for record in census.get("sites", [])}

    for key, record in found.items():
        if key not in registered:
            problems.append(
                f"unregistered claim {record['file']}:{record['line']}: "
                f"{record['anchor'][:80]}")
    for key, record in registered.items():
        if key not in found:
            problems.append(
                f"registered claim no longer present {key[0]}:{key[1]}")
            continue
        if found[key]["anchor"] != record["anchor"]:
            problems.append(
                f"anchor text changed at {key[0]}:{key[1]}")
        elif anchor_sha256(record["anchor"]) != record["anchor_sha256"]:
            problems.append(f"anchor hash stale at {key[0]}:{key[1]}")

    if receipt is not None:
        cells = {(cell["route"], cell["mechanism"]): cell["verdict"]
                 for cell in receipt["cells"]}
        for key, record in registered.items():
            token = record.get("asserted_token")
            if token is None:
                if record.get("attribution") != ATTRIBUTION_PENDING:
                    problems.append(
                        f"{key[0]}:{key[1]} claims attribution with no token")
                continue
            cell = cells.get((record.get("route"), record.get("mechanism")))
            if cell is None:
                problems.append(
                    f"{key[0]}:{key[1]} asserts a token for a cell the "
                    f"receipt does not carry: "
                    f"({record.get('route')}, {record.get('mechanism')})")
            elif cell != token:
                problems.append(
                    f"{key[0]}:{key[1]} asserts {token!r} but the receipt's "
                    f"cell says {cell!r}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve() if args.root else ri.repo_root()
    out = default_output(root)
    previous = (json.loads(out.read_text(encoding="utf-8"))
                if out.exists() else None)
    document = build_census(root, previous)
    if args.check:
        if previous is None:
            print(f"census missing: {out}", file=sys.stderr)
            return 1
        problems = check_census(root, previous)
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1 if problems else 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(document), encoding="utf-8", newline="\n")
    print(f"wrote {out} ({document['site_count']} claim lines in "
          f"{len({r['file'] for r in document['sites']})} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
