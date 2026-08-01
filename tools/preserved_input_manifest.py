"""Build and re-verify a SHA-256 manifest over a preserved set of input files.

A run input set that survives in exactly one place is one device failure away
from making every run derived from it unreproducible.  The remedy is cheap --
copy it, hash it, and write down what the hashes were -- but only if the
written-down part is checkable later.  A manifest nobody can re-verify records
a hope, not a preservation.

So this module does two jobs and keeps them apart:

``build``
    turn a *capture* -- the output of ``sha256sum`` and ``stat`` run on the
    machine that holds the files -- into a manifest.  The hashes are measured
    by the standard tool on the holding machine; this module only transcribes,
    which is why the transcription is mechanical and refuses to guess.

``verify``
    check a manifest against itself, and check a manifest against a *fresh*
    capture taken later.  Any drift -- a changed hash, a missing member, a
    location that no longer agrees with its peers -- is an error.  There is no
    tolerance to widen.

Fail-closed on replica disagreement
-----------------------------------
An input set usually exists in several directories at once.  The tempting
shortcut is to hash one of them and assume the rest match.  That assumption is
exactly the thing worth checking: if two replicas disagree, one of them has
rotted or was regenerated, and picking either silently would preserve the
wrong bytes.  :func:`build_manifest` therefore raises
:class:`PreservationConflict` listing the disagreeing rows rather than
choosing a winner.

Member naming
-------------
The same file is written with ``:`` in its timestamp by some producers and
``_`` by others, so a member is keyed on its basename with ``:`` folded to
``_``.  Each location additionally records the basename it actually carries,
so the manifest stays a faithful description of the disk.

Preserved is not the same question as admitted
----------------------------------------------
What survives and what a run may be initialized from are two decisions, and
they are not always the same set.  A file can be worth preserving precisely
because it is real history and still be the wrong input for the configuration
somebody is about to run.  A manifest that answers only the first question
leaves the second to whoever is reading it at the time.

So a manifest may carry admission sets alongside its preserved set.  A
preserved member is keyed on its member name -- one name, one set of bytes.
An admission member is keyed on the name *and* the digest, because the same
name can spell different bytes in the two sets, and where it does, naming only
the member would restate the ambiguity instead of removing it.  Each admission
member records how its bytes stand to the preserved member of that name, and
that claim is measured against the preserved member rather than believed.

The section fails closed the same way the rest of the module does: a preserved
member whose bytes no admission set admits has to be listed as not admitted,
with a written reason.  Dropping out of admission by being forgotten is the
failure this rules out.

This module names no case, no campaign and no domain: a "preserved input set"
is any set of files, and every path, host and filename in a manifest is data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import sys

SCHEMA_ID = "gpuwm.preserved-input-set/v2"

# Schema ids this module can verify.  v2 only adds the optional admission
# section; it changes nothing a v1 document already said, so v1 documents keep
# verifying unchanged and are not rewritten to gain a field they do not use.
SUPPORTED_SCHEMA_IDS = ("gpuwm.preserved-input-set/v1", SCHEMA_ID)

# Section markers a capture uses to separate one role of location from another.
# The role travels with the measurement instead of being supplied on the command
# line, so a manifest cannot be relabelled after the fact without editing the
# evidence it was built from.
SECTION_PREFIX = "### "
_SHA_SECTION = re.compile(r"^SHA256\s+(?P<role>[A-Z0-9-]+)$")
_SHA_ROW = re.compile(r"^(?P<sha>[0-9a-f]{64})\s+(?P<path>.+)$")
_STAT_ROW = re.compile(
    r"^(?P<bytes>\d+)\s+(?P<device>\d+)\s+(?P<inode>\d+)\s+"
    r"(?P<links>\d+)\s+(?P<mtime>\d+)\s+(?P<path>.+)$"
)

ROLE_ALIASES = {
    "SOURCE-LOCATIONS": "existing-replica",
    "STAGED-COPY": "preserved-copy",
}

DIGEST_KEY = "manifest_binding_sha256"

ADMISSION_KEY = "admission_sets"

# How an admission member's bytes stand to the preserved member of the same
# name.  Both values are checkable against the manifest's own member list,
# which is the whole point of writing them down: the relation is recorded as
# data and then measured, never taken on trust.
RELATION_SAME_BYTES = "same-bytes"
RELATION_DIFFERENT_BYTES = "different-bytes"
ADMISSION_RELATIONS = (RELATION_SAME_BYTES, RELATION_DIFFERENT_BYTES)


class PreservationConflict(RuntimeError):
    """Two locations claim the same member with different bytes."""


class CaptureError(ValueError):
    """A capture is not in the shape this module can transcribe."""


def canonical_member_name(basename: str) -> str:
    """Fold the one spelling difference producers introduce in timestamps."""

    return basename.replace(":", "_")


def canonical_sha256(value: object) -> str:
    """SHA-256 over a canonical JSON rendering, for binding a document."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_capture(text: str) -> dict:
    """Read a capture into ``{host, captured_utc, filesystem, rows, stats}``."""

    host = None
    captured_utc = None
    filesystem: dict[str, str] = {}
    rows: list[dict[str, str]] = []
    stats: dict[str, dict[str, int]] = {}
    section = None
    role = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(SECTION_PREFIX):
            body = line[len(SECTION_PREFIX):].strip()
            if body.startswith("capture_host "):
                host = body.split(None, 1)[1]
                continue
            if body.startswith("capture_utc "):
                captured_utc = body.split(None, 1)[1]
                continue
            sha_match = _SHA_SECTION.match(body)
            if sha_match:
                section = "sha"
                raw_role = sha_match.group("role")
                role = ROLE_ALIASES.get(raw_role, raw_role.lower())
                continue
            section = body.split()[0].lower()
            role = None
            continue
        if section == "sha":
            match = _SHA_ROW.match(line)
            if not match:
                raise CaptureError(f"unparsable sha256 row: {line!r}")
            rows.append(
                {
                    "sha256": match.group("sha"),
                    "path": match.group("path").strip(),
                    "role": role or "unlabelled",
                }
            )
            continue
        if section == "stat":
            match = _STAT_ROW.match(line)
            if match:
                stats[match.group("path").strip()] = {
                    "bytes": int(match.group("bytes")),
                    "device": int(match.group("device")),
                    "inode": int(match.group("inode")),
                    "links": int(match.group("links")),
                    "mtime": int(match.group("mtime")),
                }
            continue
        if section == "filesystem":
            parts = line.split()
            if len(parts) == 3 and "-" in parts[1]:
                filesystem = {
                    "device": parts[0],
                    "uuid": parts[1],
                    "fstype": parts[2],
                }
            continue

    if host is None or captured_utc is None:
        raise CaptureError("capture is missing capture_host or capture_utc")
    if not rows:
        raise CaptureError("capture carries no sha256 rows")
    return {
        "host": host,
        "captured_utc": captured_utc,
        "filesystem": filesystem,
        "rows": rows,
        "stats": stats,
    }


def _location_root(path: str) -> str:
    head, _, _ = path.rpartition("/")
    return head or "."


def _grouped_locations(rows: Iterable[Mapping[str, str]]) -> dict[str, dict]:
    locations: dict[str, dict] = {}
    for row in rows:
        root = _location_root(row["path"])
        basename = row["path"].rpartition("/")[2]
        member = canonical_member_name(basename)
        entry = locations.setdefault(
            root, {"root": root, "role": row["role"], "members": {}}
        )
        if member in entry["members"]:
            raise CaptureError(
                f"location {root!r} lists member {member!r} twice"
            )
        entry["members"][member] = {
            "basename": basename,
            "sha256": row["sha256"],
        }
    return locations


def build_manifest(
    capture: Mapping[str, object],
    *,
    source_of_copy: str | None = None,
    note: str | None = None,
) -> dict:
    """Transcribe a capture into a manifest, refusing on replica disagreement."""

    rows = list(capture["rows"])  # type: ignore[index]
    locations = _grouped_locations(rows)
    stats = dict(capture.get("stats") or {})  # type: ignore[union-attr]

    # Cross-location agreement is measured, never assumed.
    by_member: dict[str, dict[str, str]] = {}
    conflicts: list[str] = []
    for root, entry in locations.items():
        for member, record in entry["members"].items():
            seen = by_member.setdefault(member, {})
            seen[root] = record["sha256"]
    for member, seen in sorted(by_member.items()):
        distinct = sorted(set(seen.values()))
        if len(distinct) > 1:
            detail = ", ".join(f"{root}={sha}" for root, sha in sorted(seen.items()))
            conflicts.append(f"{member}: {detail}")
    if conflicts:
        raise PreservationConflict(
            "locations disagree on member bytes; nothing was chosen:\n  "
            + "\n  ".join(conflicts)
        )

    def _member_bytes(member: str) -> int | None:
        for path, stat in stats.items():
            if canonical_member_name(path.rpartition("/")[2]) == member:
                return stat["bytes"]
        return None

    members = []
    for member in sorted(by_member):
        size = _member_bytes(member)
        members.append(
            {
                "member": member,
                "sha256": next(iter(by_member[member].values())),
                "bytes": size,
                "bytes_status": "measured" if size is not None else "unavailable",
            }
        )

    location_records = []
    for root in sorted(locations):
        entry = locations[root]
        role = entry["role"]
        if source_of_copy is not None and root == source_of_copy:
            role = "source-of-copy"
        location_records.append(
            {
                "root": root,
                "role": role,
                "members": {
                    member: entry["members"][member]["basename"]
                    for member in sorted(entry["members"])
                },
            }
        )

    devices = sorted({stat["device"] for stat in stats.values()})
    total = sum(m["bytes"] for m in members if m["bytes"] is not None)

    manifest: dict[str, object] = {
        "schema": SCHEMA_ID,
        "captured_utc": capture["captured_utc"],
        "host": capture["host"],
        "filesystem": capture.get("filesystem") or {},
        "distinct_devices": devices,
        "member_count": len(members),
        "total_bytes": total,
        "members": members,
        "locations": location_records,
    }
    if note:
        manifest["note"] = note
    manifest[DIGEST_KEY] = canonical_sha256(manifest)
    return manifest


def _admission_members(entry: Mapping[str, object], label: str) -> tuple[list, list[str]]:
    """Validate one admission set's own member rows.  Returns ``(rows, issues)``."""

    issues: list[str] = []
    rows = list(entry.get("members") or [])
    if not rows:
        issues.append(f"{label}: admits nothing")

    names = [row.get("member") for row in rows]
    if len(names) != len(set(names)):
        issues.append(f"{label}: the same member is admitted more than once")
    for row in rows:
        name = row.get("member")
        if name != canonical_member_name(str(name)):
            issues.append(f"{label}: {name!r} is not a canonical member name")
        if not re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256", ""))):
            issues.append(f"{label}: {name!r}: sha256 is not 64 lowercase hex")
        if row.get("bytes_status") == "measured" and not isinstance(
            row.get("bytes"), int
        ):
            issues.append(f"{label}: {name!r}: bytes measured but not an integer")

    declared = entry.get("member_count")
    if declared is not None and declared != len(rows):
        issues.append(
            f"{label}: member_count is {declared!r}, {len(rows)} members are listed"
        )
    measured = [
        row
        for row in rows
        if row.get("bytes_status") == "measured" and isinstance(row.get("bytes"), int)
    ]
    total = entry.get("total_bytes")
    if total is not None and measured and total != sum(row["bytes"] for row in measured):
        issues.append(
            f"{label}: total_bytes does not equal the sum of measured member bytes"
        )
    return rows, issues


def verify_admission_sets(manifest: Mapping[str, object]) -> list[str]:
    """Check the admission sets a manifest carries.  Returns the issues found.

    Every admitted member is held against the preserved member of the same
    name: a row claiming its bytes are the preserved ones must hash what that
    member hashes, and a row claiming they are not must hash something else and
    say where its own bytes live, because the preserved set's locations do not
    cover them.  A row naming a member the preserved set has never heard of is
    an error rather than a silent addition -- admitting bytes nothing preserved
    is the situation this whole module exists to prevent.

    Then the other direction, which is the one that fails closed: a preserved
    member whose exact bytes no set admits must appear in that set's
    ``not_admitted`` list with a written reason.  Preserving a file and
    declining to run from it is a legitimate, ordinary decision; leaving it out
    without saying so is indistinguishable from having forgotten it.
    """

    issues: list[str] = []
    sets = manifest.get(ADMISSION_KEY)
    if sets is None:
        return issues
    if manifest.get("schema") != SCHEMA_ID:
        issues.append(
            f"{ADMISSION_KEY} is a {SCHEMA_ID} feature; this document declares "
            f"{manifest.get('schema')!r}"
        )
    if not isinstance(sets, list) or not sets:
        issues.append(f"{ADMISSION_KEY} is present but names no set")
        return issues

    preserved = {
        member.get("member"): member.get("sha256")
        for member in manifest.get("members") or []
    }

    identifiers = [entry.get("id") for entry in sets]
    if len(identifiers) != len(set(identifiers)):
        issues.append(f"{ADMISSION_KEY} names the same id more than once")

    for entry in sets:
        identifier = entry.get("id")
        label = repr(identifier)
        if not str(identifier or "").strip():
            issues.append(f"{ADMISSION_KEY}: an admission set carries no id")
        if not str(entry.get("note") or "").strip():
            issues.append(f"{label}: admits without saying in writing what it admits")

        rows, row_issues = _admission_members(entry, label)
        issues.extend(row_issues)

        admitted: set[tuple[object, str]] = set()
        for row in rows:
            name = row.get("member")
            sha = str(row.get("sha256", ""))
            admitted.add((name, sha))
            relation = row.get("preserved_member_relation")
            if relation not in ADMISSION_RELATIONS:
                issues.append(
                    f"{label}: {name!r}: preserved_member_relation is "
                    f"{relation!r}, expected one of {list(ADMISSION_RELATIONS)}"
                )
                continue
            if name not in preserved:
                issues.append(
                    f"{label}: {name!r}: no preserved member carries this name, so "
                    "its relation to one cannot hold"
                )
                continue
            if relation == RELATION_SAME_BYTES and preserved[name] != sha:
                issues.append(
                    f"{label}: {name!r}: claims {RELATION_SAME_BYTES} but the "
                    f"preserved member hashes {preserved[name]}, not {sha}"
                )
            if relation == RELATION_DIFFERENT_BYTES:
                if preserved[name] == sha:
                    issues.append(
                        f"{label}: {name!r}: claims {RELATION_DIFFERENT_BYTES} but "
                        "hashes exactly what the preserved member of that name does"
                    )
                elif not str(row.get("root") or "").strip():
                    issues.append(
                        f"{label}: {name!r}: differs from the preserved member of "
                        "that name, so the preserved locations do not hold it, and "
                        "it records no root of its own"
                    )

        excluded: set[tuple[object, str]] = set()
        for row in entry.get("not_admitted") or []:
            key = (row.get("member"), str(row.get("sha256", "")))
            excluded.add(key)
            if key in admitted:
                issues.append(
                    f"{label}: {key[0]!r} is admitted and recorded as not admitted "
                    "at the same time"
                )
            if not str(row.get("reason") or "").strip():
                issues.append(
                    f"{label}: {key[0]!r} is recorded as not admitted with no "
                    "written reason"
                )
        for name, sha in sorted(preserved.items(), key=lambda item: str(item[0])):
            if (name, sha) in admitted or (name, sha) in excluded:
                continue
            issues.append(
                f"{label}: preserved member {name!r} ({sha}) is neither admitted "
                "nor recorded as not admitted"
            )

    return issues


def verify_manifest(manifest: Mapping[str, object]) -> list[str]:
    """Check a manifest against itself.  Returns the issues found."""

    issues: list[str] = []
    if manifest.get("schema") not in SUPPORTED_SCHEMA_IDS:
        issues.append(
            f"schema is {manifest.get('schema')!r}, expected one of "
            f"{list(SUPPORTED_SCHEMA_IDS)}"
        )

    bound = {k: v for k, v in manifest.items() if k != DIGEST_KEY}
    recorded = manifest.get(DIGEST_KEY)
    if recorded != canonical_sha256(bound):
        issues.append("manifest_binding_sha256 does not match the document it binds")

    members = manifest.get("members") or []
    if not members:
        issues.append("manifest carries no members")
    names = [m.get("member") for m in members]
    if len(names) != len(set(names)):
        issues.append("duplicate member names")
    for member in members:
        sha = str(member.get("sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", sha):
            issues.append(f"{member.get('member')!r}: sha256 is not 64 lowercase hex")
        if member.get("bytes_status") == "measured" and not isinstance(
            member.get("bytes"), int
        ):
            issues.append(f"{member.get('member')!r}: bytes measured but not an integer")

    expected = {m.get("member"): m.get("sha256") for m in members}
    locations = manifest.get("locations") or []
    if not locations:
        issues.append("manifest carries no locations")
    for location in locations:
        held = set(location.get("members") or {})
        missing = sorted(set(expected) - held)
        extra = sorted(held - set(expected))
        if missing:
            issues.append(f"{location.get('root')!r}: missing members {missing}")
        if extra:
            issues.append(f"{location.get('root')!r}: unknown members {extra}")

    roles = {location.get("role") for location in locations}
    if "preserved-copy" not in roles:
        issues.append("no location carries role 'preserved-copy'")

    measured = [m for m in members if m.get("bytes_status") == "measured"]
    if measured and manifest.get("total_bytes") != sum(m["bytes"] for m in measured):
        issues.append("total_bytes does not equal the sum of measured member bytes")

    issues.extend(verify_admission_sets(manifest))

    return issues


def verify_against_capture(
    manifest: Mapping[str, object], capture: Mapping[str, object]
) -> list[str]:
    """Re-verify a manifest against a capture taken later.  Any drift is an issue."""

    issues = verify_manifest(manifest)
    expected = {m["member"]: m["sha256"] for m in manifest.get("members") or []}
    fresh = _grouped_locations(capture["rows"])  # type: ignore[index]

    checked_any = False
    for location in manifest.get("locations") or []:
        root = location.get("root")
        if root not in fresh:
            issues.append(f"{root!r}: absent from the capture")
            continue
        checked_any = True
        held = fresh[root]["members"]
        for member, sha in sorted(expected.items()):
            if member not in held:
                issues.append(f"{root!r}: member {member!r} is gone")
                continue
            if held[member]["sha256"] != sha:
                issues.append(
                    f"{root!r}: member {member!r} hashes "
                    f"{held[member]['sha256']}, manifest records {sha}"
                )
    if not checked_any:
        issues.append("the capture covers none of the manifest's locations")
    return issues


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--capture", type=Path, help="capture to build a manifest from")
    parser.add_argument("--out", type=Path, help="where to write the built manifest")
    parser.add_argument("--source-of-copy", help="root the preserved copy was taken from")
    parser.add_argument("--note", help="free-text note carried in the manifest")
    parser.add_argument("--verify", type=Path, help="manifest to verify")
    parser.add_argument(
        "--against-capture", type=Path, help="fresh capture to re-verify the manifest against"
    )
    args = parser.parse_args(argv)

    if args.capture is not None:
        capture = parse_capture(args.capture.read_text(encoding="utf-8"))
        manifest = build_manifest(
            capture, source_of_copy=args.source_of_copy, note=args.note
        )
        rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        if args.out is not None:
            args.out.write_text(rendered, encoding="utf-8")
        else:
            sys.stdout.write(rendered)

    if args.verify is not None:
        manifest = _load_json(args.verify)
        if args.against_capture is not None:
            capture = parse_capture(args.against_capture.read_text(encoding="utf-8"))
            issues = verify_against_capture(manifest, capture)
        else:
            issues = verify_manifest(manifest)
        for issue in issues:
            print(f"FAIL {issue}")
        if issues:
            return 1
        admitted = "; ".join(
            f"{entry.get('id')} admits {len(entry.get('members') or [])}"
            for entry in manifest.get(ADMISSION_KEY) or []
        )
        detail = f"; {admitted}" if admitted else ""
        print(
            f"OK {args.verify}: {manifest['member_count']} members verified{detail}"
        )

    if args.capture is None and args.verify is None:
        parser.error("nothing to do: pass --capture or --verify")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
