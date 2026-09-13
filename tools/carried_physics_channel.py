#!/usr/bin/env python3
"""Create, receive and verify carried-physics release changes without a GPU.

The transport contains scoped raw engine snapshots. Release hunks are diagnostic
old/new byte ranges, never receiver divergence identities. A trusted local
receiver supplies the mapping, normalization and classification join. Nothing
is merged, installed, published, or written over an existing output.
"""
from __future__ import annotations

import argparse
import base64
import binascii
from collections import Counter, defaultdict
from copy import deepcopy
from difflib import SequenceMatcher
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import stat
import sys
from types import ModuleType
from typing import Any

SCOPE_SCHEMA = "arwen.carried-scope.v1"
RELEASE_SCHEMA = "arwen.carried-release.v2"
REVIEW_SCHEMA = "arwen.carried-review.v2"
FEEDBACK_SCHEMA = "arwen.carried-feedback.v2"
CLASSIFIED_SCHEMA = "arwen.carried-classified.v2"
DECISIONS_SCHEMA = "arwen.carried-decisions.v2"
ROW_FIELDS = ("file", "engine_file", "engine_lines", "carried_lines",
              "engine_sha256", "carried_sha256")
DECISION_FIELDS = ("row", "class", "decision", "note")


class ChannelError(ValueError):
    """Invalid or incomplete evidence, with no source tree modified."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def seal(value: dict) -> dict:
    out = dict(value)
    out.pop("id", None)
    out["id"] = digest(canonical(out))
    return out


def checked(value: dict, schema: str) -> None:
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ChannelError(f"expected {schema}; regenerate with a compatible tool")
    if value.get("id") != seal(value)["id"]:
        raise ChannelError(f"{schema}: content id mismatch; obtain the intact artifact")


def _unique_json(pairs: list[tuple[str, Any]]) -> dict:
    out = {}
    for key, value in pairs:
        if key in out:
            raise ChannelError(f"duplicate JSON key {key!r}")
        out[key] = value
    return out


def read_json(path: Path) -> dict:
    def reject(value):
        raise ChannelError(f"nonfinite JSON value: {value}")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json,
                      parse_constant=reject)


def write_new(path: Path, payload: dict) -> None:
    """Exclusive creation, including when another writer wins the race."""
    data = (json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode()
    try:
        with path.open("xb") as handle:
            handle.write(data)
    except FileExistsError as exc:
        raise ChannelError(f"output already exists: {path}; choose a new output path") from exc


def safe_path(value: str) -> str:
    if (not isinstance(value, str) or not value or value.startswith("/")
            or "\\" in value or ":" in value
            or any(ord(ch) < 32 or ord(ch) == 127 for ch in value)
            or any(0xD800 <= ord(ch) <= 0xDFFF for ch in value)
            or any(part in ("", ".", "..") for part in value.split("/"))):
        raise ChannelError(f"not a canonical relative file path: {value!r}")
    return value


def within(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix + "/")


def validate_scope(scope: dict) -> None:
    if not isinstance(scope, dict) or scope.get("schema") != SCOPE_SCHEMA:
        raise ChannelError("missing carried-scope v1 metadata; export the receiver scope")
    if set(scope) != {"schema", "engine_root", "units"}:
        raise ChannelError("scope has unexpected fields")
    root = safe_path(scope.get("engine_root"))
    units = scope.get("units")
    if not isinstance(units, list) or not units:
        raise ChannelError("scope has no units; export the complete receiver mapping")
    for i, unit in enumerate(units):
        if not isinstance(unit, dict) or set(unit) != {"engine", "carried", "kind"}:
            raise ChannelError(f"scope unit {i} needs engine, carried and kind")
        for field in ("engine", "carried"):
            safe_path(unit[field])
            for earlier in units[:i]:
                if within(unit[field], earlier[field]) or within(earlier[field], unit[field]):
                    raise ChannelError(f"overlapping {field} mappings at {unit[field]}")
        if not unit["engine"].startswith(root + "/"):
            raise ChannelError(f"{unit['engine']} is outside engine root {root}")
        if unit["kind"] not in ("file", "tree"):
            raise ChannelError(f"scope kind must be file or tree: {unit}")


def owner(scope: dict, path: str, field: str = "engine") -> dict:
    safe_path(path)
    for unit in scope["units"]:
        if (path == unit[field] if unit["kind"] == "file"
                else path.startswith(unit[field] + "/")):
            return unit
    raise ChannelError(f"{path} is not a file in the explicit carried scope")


def remap(scope: dict, path: str, field: str = "engine") -> str:
    unit = owner(scope, path, field)
    target = "carried" if field == "engine" else "engine"
    return unit[target] + path[len(unit[field]):]


def blob(raw: bytes | None) -> dict | None:
    if raw is None:
        return None
    return {"sha256": digest(raw), "size": len(raw),
            "base64": base64.b64encode(raw).decode("ascii")}


def unblob(value: dict | None) -> bytes | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"sha256", "size", "base64"}:
        raise ChannelError("blob needs sha256, size and base64, or null for absence")
    try:
        raw = base64.b64decode(value["base64"], validate=True)
    except (ValueError, TypeError, binascii.Error) as exc:
        raise ChannelError("invalid base64 blob") from exc
    if (type(value["size"]) is not int or value["size"] != len(raw)
            or value["sha256"] != digest(raw)):
        raise ChannelError("raw blob hash or size mismatch")
    if base64.b64encode(raw).decode("ascii") != value["base64"]:
        raise ChannelError("noncanonical base64 blob")
    return raw


def release_hunks(before: bytes | None, after: bytes | None) -> list[dict]:
    """Raw byte offsets, half-open, with distinct old/new-side hashes.

    This instrument deliberately has no receiver normalization. For UTF-8,
    splitlines(keepends=True) supplies byte lines so CRLF and EOF edits remain
    visible. Invalid UTF-8 uses one whole-file byte range. These are not keys
    suitable for joining classifications.
    """
    if before == after:
        return []
    try:
        for raw in (before, after):
            if raw is not None:
                raw.decode("utf-8")
        lines = [raw.splitlines(keepends=True) if raw is not None else []
                 for raw in (before, after)]
        offsets = []
        for side in lines:
            positions = [0]
            for line in side:
                positions.append(positions[-1] + len(line))
            offsets.append(positions)
        ranges = [(offsets[0][i], offsets[0][j], offsets[1][k], offsets[1][l])
                  for tag, i, j, k, l in SequenceMatcher(
                      None, lines[0], lines[1], autojunk=False).get_opcodes()
                  if tag != "equal"]
        kind = "raw-utf8-byte-ranges"
    except UnicodeDecodeError:
        ranges = [(0, len(before or b""), 0, len(after or b""))]
        kind = "raw-binary-byte-ranges"
    # Adding/removing an empty file has no line opcode but changes existence.
    if not ranges:
        ranges = [(0, 0, 0, 0)]
    return [{"kind": kind, "old_bytes": [i, j], "new_bytes": [k, l],
             "old_sha256": digest((before or b"")[i:j]),
             "new_sha256": digest((after or b"")[k:l])}
            for i, j, k, l in ranges]


def git(repo: Path, *args: str) -> bytes:
    # No shell, archive extraction, textconv, hooks or executable file content.
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True)
    if proc.returncode:
        raise ChannelError(f"git {' '.join(args[:2])} failed: "
                           + proc.stderr.decode("utf-8", errors="replace").strip())
    return proc.stdout


class GitTree:
    """Immutable Git objects, never dirty worktree bytes carrying a clean SHA."""

    def __init__(self, repo: Path, revision: str, version: str):
        self.repo = Path(repo)
        self.commit = git(self.repo, "rev-parse", "--verify", "--end-of-options",
                          revision + "^{commit}").decode().strip()
        self.tree = git(self.repo, "rev-parse", self.commit + "^{tree}").decode().strip()
        self.meta = {"commit": self.commit, "tree": self.tree,
                     "committed_utc": datetime.fromtimestamp(int(git(self.repo, "show",
                         "-s", "--format=%ct", self.commit)), timezone.utc).isoformat(),
                     "version": version}
        self.entries = {}
        for line in git(self.repo, "ls-tree", "-rz", "--full-tree", self.commit).split(b"\0"):
            if not line:
                continue
            info, raw_path = line.split(b"\t", 1)
            mode, kind, oid = info.decode("ascii").split()
            # Only scoped names are validated/read below. Unrelated repository
            # files, including private code, never enter the artifact.
            path = raw_path.decode("utf-8", errors="surrogateescape")
            self.entries[path] = (mode, kind, oid)

    def snapshot(self, scope: dict) -> dict[str, bytes | None]:
        validate_scope(scope)
        out = {}
        for unit in scope["units"]:
            source = unit["engine"]
            # An ancestor symlink/gitlink cannot be mistaken for absent files.
            for parent in PurePosixPath(source).parents:
                if str(parent) in self.entries:
                    raise ChannelError(f"mapped path has a non-directory ancestor: {source}")
            children = [p for p in self.entries if p.startswith(source + "/")]
            if unit["kind"] == "tree":
                if source in self.entries:
                    raise ChannelError(f"mapped tree is not a directory: {source}")
                names = sorted(children)
            else:
                if children:
                    raise ChannelError(f"mapped file is a directory: {source}")
                names = [source]
            for path in names:
                safe_path(path)
                entry = self.entries.get(path)
                if entry is None:
                    out[path] = None
                    continue
                mode, kind, oid = entry
                if kind != "blob" or mode not in ("100644", "100755"):
                    raise ChannelError(f"unsupported carried object {path}: {mode} {kind}; "
                                       "provide a regular file rather than following a link")
                out[path] = git(self.repo, "cat-file", "blob", oid)
        return out


def build_release(scope: dict, before: dict[str, bytes | None],
                  after: dict[str, bytes | None], old: dict, new: dict) -> dict:
    """Pure API. Caller-supplied provenance is independently checked by verify-git."""
    validate_scope(scope)
    if set(before) != set(after):
        raise ChannelError("snapshots need the same explicit inventory; None alone means absence")
    paths = set(before)
    required = {u["engine"] for u in scope["units"] if u["kind"] == "file"}
    if not required <= paths:
        raise ChannelError("snapshots omit a mapped file; supply None only for proven absence")
    for label, meta in (("old", old), ("new", new)):
        if (not isinstance(meta, dict) or
                set(meta) != {"commit", "tree", "committed_utc", "version"} or
                any(not isinstance(meta.get(k), str) or not meta[k] for k in meta) or
                any(re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", meta[k]) is None
                    for k in ("commit", "tree"))):
            raise ChannelError(f"{label} snapshot needs full commit/tree ids, a timestamp and version label")
        try:
            stamp = datetime.fromisoformat(meta["committed_utc"])
        except ValueError as exc:
            raise ChannelError(f"{label} snapshot timestamp is not ISO 8601") from exc
        if stamp.utcoffset() is None:
            raise ChannelError(f"{label} snapshot timestamp needs an explicit timezone")
    files = []
    for path in sorted(paths):
        owner(scope, path)
        a, b = before[path], after[path]
        if any(raw is not None and not isinstance(raw, bytes) for raw in (a, b)):
            raise ChannelError(f"{path}: snapshot values must be bytes or None")
        hunks = release_hunks(a, b)
        for i, hunk in enumerate(hunks):
            hunk["id"] = digest(canonical({"namespace": "engine-old-new",
                "path": path, "old": old, "new": new, "ordinal": i, "hunk": hunk}))
        files.append({"path": path, "carried": remap(scope, path),
                      "before": blob(a), "after": blob(b),
                      "changed": a != b, "release_hunks": hunks})
    return seal({"schema": RELEASE_SCHEMA, "scope": deepcopy(scope),
                 "old": deepcopy(old), "new": deepcopy(new), "files": files})


def unpack_release(manifest: dict) -> tuple[dict, dict]:
    checked(manifest, RELEASE_SCHEMA)
    validate_scope(manifest["scope"])
    if not isinstance(manifest.get("files"), list):
        raise ChannelError("release files must be a complete list")
    before, after = {}, {}
    for entry in manifest["files"]:
        if not isinstance(entry, dict):
            raise ChannelError("release file entry must be an object")
        path = safe_path(entry["path"])
        if path in before:
            raise ChannelError(f"duplicate release file: {path}")
        before[path], after[path] = unblob(entry["before"]), unblob(entry["after"])
    expected = build_release(manifest["scope"], before, after,
                             manifest["old"], manifest["new"])
    if expected != manifest:
        raise ChannelError("release contents, scope coverage or hunk descriptors disagree")
    return before, after


def create_release(repo: Path, old_ref: str, new_ref: str, scope: dict,
                   old_version: str, new_version: str, *, old_repo: Path | None = None) -> dict:
    old, new = GitTree(old_repo or repo, old_ref, old_version), GitTree(repo, new_ref, new_version)
    before, after = old.snapshot(scope), new.snapshot(scope)
    # Both Git inventories have been enumerated. Only here may an absent member
    # be completed with None. An arbitrary partial mapping is not such proof.
    paths = set(before) | set(after)
    return build_release(scope, {p: before.get(p) for p in paths},
                         {p: after.get(p) for p in paths}, old.meta, new.meta)


def verify_git(manifest: dict, repo: Path, *, old_repo: Path | None = None) -> None:
    """Check coverage and every blob independently against immutable Git objects."""
    unpack_release(manifest)
    actual = create_release(repo, manifest["old"]["commit"], manifest["new"]["commit"],
                            manifest["scope"], manifest["old"]["version"],
                            manifest["new"]["version"], old_repo=old_repo)
    if actual != manifest:
        raise ChannelError("manifest differs from the named Git snapshots; "
                           "regenerate from the intended committed revisions")


def load_receiver(path: Path) -> ModuleType:
    """Load only a caller-selected local tool with its own local rule module.

    Module-cache isolation is synchronous; callers must serialize tool loading.
    No filename or executable code is read from an incoming JSON artifact.
    """
    path = path.resolve()
    saved_path = list(sys.path)
    previous = sys.modules.pop("resync_from_owner", None)
    try:
        sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location(
            "_carried_receiver_" + digest(str(path).encode())[:16], path)
        if spec is None or spec.loader is None:
            raise ChannelError(f"cannot load local receiver: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for method in ("channel_scope", "channel_protocol", "hunks_from_bytes",
                       "rejoin_rows", "divergence_document", "key"):
            if not callable(getattr(module, method, None)):
                raise ChannelError(f"receiver lacks {method}; install its channel API extension")
        if module.channel_protocol().get("byte_api") != "named-raw-sides.v1":
            raise ChannelError("receiver must accept named engine_raw and carried_raw arguments")
        return module
    finally:
        sys.path[:] = saved_path
        sys.modules.pop("resync_from_owner", None)
        if previous is not None:
            sys.modules["resync_from_owner"] = previous


def _is_link(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        return bool(getattr(path.lstat(), "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    except FileNotFoundError:
        return False


def read_carried(scope: dict, root: Path) -> dict[str, bytes | None]:
    validate_scope(scope)
    root = root.resolve()
    if not root.is_dir():
        raise ChannelError(f"carried package root does not exist: {root}")
    out = {}
    for unit in scope["units"]:
        relative = unit["carried"]
        path = root / relative
        for part in (path, *path.parents):
            if part == root:
                break
            if _is_link(part):
                raise ChannelError(f"refusing carried symlink: {part}")
            if part != path and part.exists() and not part.is_dir():
                raise ChannelError(f"carried path has a non-directory ancestor: {relative}")
        if unit["kind"] == "file":
            if path.exists() and not path.is_file():
                raise ChannelError(f"mapped carried file is not regular: {relative}")
            out[relative] = path.read_bytes() if path.is_file() else None
        else:
            if path.exists() and not path.is_dir():
                raise ChannelError(f"mapped carried tree is not a directory: {relative}")
            if not path.exists():
                continue
            def walk_error(error):
                raise ChannelError(f"cannot enumerate carried tree {relative}: {error}")
            for directory, dirs, names in os.walk(path, followlinks=False, onerror=walk_error):
                for name in dirs + names:
                    item = Path(directory) / name
                    if _is_link(item):
                        raise ChannelError(f"refusing carried symlink: {item}")
                for name in names:
                    item = Path(directory) / name
                    if not item.is_file():
                        raise ChannelError(f"carried object is not regular: {item}")
                    rel = item.relative_to(root).as_posix()
                    safe_path(rel)
                    out[rel] = item.read_bytes()
    return out


def descriptor(raw: bytes | None) -> dict | None:
    if raw is not None and not isinstance(raw, bytes):
        raise ChannelError("raw file contents must be bytes or explicit absence")
    return None if raw is None else {"sha256": digest(raw), "size": len(raw)}


def require_release(manifest: dict, expected_release_id: str) -> tuple[dict, dict]:
    """The expected id must come from local emission or an authenticated channel.

    A received object's own id is not an independent trust input. This check
    never asserts that its source is scientifically correct.
    """
    before, after = unpack_release(manifest)
    if not isinstance(expected_release_id, str) or expected_release_id != manifest["id"]:
        raise ChannelError("release differs from the independently trusted release id")
    return before, after


def _agree_receiver(manifest: dict, receiver: ModuleType) -> None:
    if receiver.channel_scope() != manifest["scope"]:
        raise ChannelError("release scope differs from the receiver; regenerate from its scope")
    if receiver.channel_protocol().get("byte_api") != "named-raw-sides.v1":
        raise ChannelError("receiver must implement named raw sides")


def _carried_identity(scope: dict, carried: dict) -> dict:
    required = {u["carried"] for u in scope["units"] if u["kind"] == "file"}
    if not required <= set(carried):
        raise ChannelError("carried snapshot omits a mapped file; supply explicit absence")
    for rel, raw in carried.items():
        owner(scope, rel, "carried")
        if raw is not None and not isinstance(raw, bytes):
            raise ChannelError(f"{rel}: carried contents must be bytes or explicit absence")
    return {p: descriptor(carried[p]) for p in sorted(carried)}


def pair_rows(scope: dict, engine: dict, carried: dict, receiver: ModuleType) -> list[dict]:
    """Measure a complete snapshot in the receiver's mapping and occurrence order.

    Missing engine directory members are known absent only after authenticating
    the complete release inventory. Missing carried directory members assume a
    complete local snapshot, as returned by read_carried, not a partial map.
    """
    _carried_identity(scope, carried)
    paths = set(engine) | {remap(scope, p, "carried") for p in carried}
    for p in paths:
        owner(scope, p)
    rows = []
    for unit in scope["units"]:
        names = ([unit["engine"]] if unit["kind"] == "file" else
                 sorted(p for p in paths if p.startswith(unit["engine"] + "/")))
        for path in names:
            rel = remap(scope, path)
            engine_rel = str(PurePosixPath(path).relative_to(scope["engine_root"]))
            rows.extend(receiver.hunks_from_bytes(rel, engine_rel,
                        engine_raw=engine.get(path), carried_raw=carried.get(rel)))
    return rows


def _multiset(rows: list[dict], receiver: ModuleType) -> Counter:
    return Counter((r["engine_file"], *receiver.key(r)) for r in rows)


def review_release(manifest: dict, receiver: ModuleType, carried: dict,
                   previous: dict | None = None, *, expected_release_id: str) -> dict:
    """Recompute both pairs before inheriting any previous annotations."""
    before, after = require_release(manifest, expected_release_id)
    _agree_receiver(manifest, receiver)
    identity = _carried_identity(manifest["scope"], carried)
    protocol = receiver.channel_protocol()
    baseline = pair_rows(manifest["scope"], before, carried, receiver)
    measured = pair_rows(manifest["scope"], after, carried, receiver)
    prior = []
    if previous is not None:
        if not isinstance(previous, dict):
            raise ChannelError("previous registry must be an object")
        if (previous.get("schema") != protocol["row_schema"] or
                previous.get("engine_version") != manifest["old"]["version"]):
            raise ChannelError("previous registry schema/version is not the release baseline")
        prior = previous["rows"]
        if _multiset(prior, receiver) != _multiset(baseline, receiver):
            raise ChannelError("previous rows do not describe the actual old-engine/current-carried "
                               "pair under these rules; remeasure or explicitly bootstrap")
    rows, retired = receiver.rejoin_rows(measured, prior)
    counts = Counter(receiver.key(r) for r in prior)
    fresh = []
    for i, row in enumerate(measured):
        key = receiver.key(row)
        if counts[key]:
            counts[key] -= 1
        else:
            fresh.append(i)
    choices = defaultdict(set)
    for row in prior:
        choices[receiver.key(row)].add(tuple(row.get(f, "") for f in DECISION_FIELDS))
    current_keys = {receiver.key(r) for r in measured}
    ambiguous = [list(k) for k, values in choices.items()
                 if len(values) > 1 and k in current_keys]
    return seal({"schema": REVIEW_SCHEMA, "release_id": manifest["id"],
        "protocol": protocol, "carried_identity": identity,
        "previous_registry": deepcopy(previous),
        "baseline_validation": "bootstrap-no-inheritance" if previous is None else
                               "exact-path-and-key-multiset-under-recorded-rules",
        "baseline_rows": baseline,
        "candidate": receiver.divergence_document(manifest["new"]["version"], rows),
        "retired_rows": retired, "new_row_indices": fresh, "ambiguous_keys": ambiguous,
        "changed_engine_files": [f["path"] for f in manifest["files"] if f["changed"]],
        "needs_review": bool(fresh or retired or ambiguous or
            any(not r.get("row") or r.get("class") in (None, "", "unknown") or
                not r.get("decision") or not r.get("note") for r in rows))})


def check_review(manifest: dict, receiver: ModuleType, carried: dict, report: dict,
                 *, expected_release_id: str) -> None:
    """Rebuild measured fields, retirements, ambiguity and counts, not just a seal."""
    checked(report, REVIEW_SCHEMA)
    actual = review_release(manifest, receiver, carried, report["previous_registry"],
                            expected_release_id=expected_release_id)
    if report != actual:
        raise ChannelError("review no longer matches the release, carried bytes, rules or baseline")


def _annotation(row: dict, protocol: dict) -> None:
    if any(not isinstance(row.get(f), str) or not row[f].strip() for f in DECISION_FIELDS):
        raise ChannelError("classification needs nonempty row, class, decision and note")
    if re.fullmatch(r"[A-Z][A-Z0-9-]*-\d+", row["row"]) is None:
        raise ChannelError("classification needs a document row identifier")
    if row["class"] == "unknown" or row["class"] not in protocol["classes"]:
        raise ChannelError("classification is unknown or unsupported")
    if row["decision"] not in protocol["decisions"]:
        raise ChannelError("unsupported classification decision")


def classify_review(report: dict, decisions: dict, receiver: ModuleType, *,
                    expected_review_id: str) -> dict:
    """Retain explicit decisions and every acknowledged retirement in a receipt."""
    checked(report, REVIEW_SCHEMA)
    if expected_review_id != report["id"]:
        raise ChannelError("decisions require the independently selected review id")
    protocol = receiver.channel_protocol()
    if report["protocol"] != protocol:
        raise ChannelError("review rules or receiver changed; remeasure before classifying")
    if (not isinstance(decisions, dict) or
            set(decisions) != {"schema", "review_id", "acknowledged_retired", "rows"} or
            decisions["schema"] != DECISIONS_SCHEMA or decisions["review_id"] != report["id"]):
        raise ChannelError("decisions must identify this exact review and its retirement list")
    acknowledged = decisions["acknowledged_retired"]
    if (not isinstance(acknowledged, list) or
            any(type(i) is not int for i in acknowledged) or
            sorted(acknowledged) != list(range(len(report["retired_rows"])))):
        raise ChannelError("acknowledge every retired row exactly once and update its document entry")
    rows = deepcopy(report["candidate"]["rows"])
    annotations = decisions["rows"]
    if not isinstance(annotations, list):
        raise ChannelError("decisions rows must be indexed annotations")
    seen = set()
    for entry in annotations:
        if not isinstance(entry, dict) or set(entry) != {"index", *DECISION_FIELDS}:
            raise ChannelError("annotations may edit only row, class, decision and note")
        i = entry["index"]
        if type(i) is not int or i < 0 or i >= len(rows) or i in seen:
            raise ChannelError("decision index is absent or repeated")
        _annotation(entry, protocol)
        seen.add(i)
        rows[i].update({f: entry[f] for f in DECISION_FIELDS})
    ambiguous = {tuple(k) for k in report["ambiguous_keys"]}
    for i, row in enumerate(rows):
        if receiver.key(row) in ambiguous and i not in seen:
            raise ChannelError(f"row index {i} has conflicting duplicate history; decide this occurrence")
        _annotation(row, protocol)
    return seal({"schema": CLASSIFIED_SCHEMA, "review": deepcopy(report),
                 "decisions": deepcopy(decisions),
                 "candidate": receiver.divergence_document(report["candidate"]["engine_version"], rows)})


def check_classified(receipt: dict, receiver: ModuleType) -> None:
    checked(receipt, CLASSIFIED_SCHEMA)
    if receipt != classify_review(receipt["review"], receipt["decisions"], receiver,
                                  expected_review_id=receipt["review"]["id"]):
        raise ChannelError("classified receipt disagrees with its explicit decisions")


def candidate_document(manifest: dict, receiver: ModuleType, carried: dict,
                       receipt: dict, *, expected_release_id: str) -> dict:
    check_classified(receipt, receiver)
    check_review(manifest, receiver, carried, receipt["review"],
                 expected_release_id=expected_release_id)
    return receiver.divergence_document(receipt["candidate"]["engine_version"],
                                       deepcopy(receipt["candidate"]["rows"]))


def make_feedback(manifest: dict, receiver: ModuleType, carried: dict,
                  receipt: dict, selected: list[int], evidence: list[str], *,
                  expected_release_id: str, include_sources: list[str] | None = None) -> dict:
    """Offer named occurrences. Full file context requires explicit path selection."""
    candidate = candidate_document(manifest, receiver, carried, receipt,
                                   expected_release_id=expected_release_id)
    rows = candidate["rows"]
    if (not selected or any(type(i) is not int or not 0 <= i < len(rows) for i in selected)
            or len(set(selected)) != len(selected)):
        raise ChannelError("feedback needs distinct in-range review indexes")
    if (not isinstance(evidence, list) or not evidence or
            any(not isinstance(e, str) or not e.strip() for e in evidence)):
        raise ChannelError("feedback needs explicit behavioral evidence references")
    chosen = []
    counts = Counter()
    for i, row in enumerate(rows):
        ordinal = counts[row["file"]]
        counts[row["file"]] += 1
        if i in selected:
            if row["decision"] != "offer":
                raise ChannelError("feedback requires a classified offer")
            chosen.append({"review_index": i, "file_ordinal": ordinal, "row": deepcopy(row)})
    paths = {item["row"]["file"] for item in chosen}
    include_sources = [] if include_sources is None else include_sources
    if (not isinstance(include_sources, list) or any(not isinstance(p, str) for p in include_sources)
            or len(set(include_sources)) != len(include_sources) or not set(include_sources) <= paths):
        raise ChannelError("source context may name each selected finding file once, and no other file")
    return seal({"schema": FEEDBACK_SCHEMA, "release_id": manifest["id"],
        "review_id": receipt["review"]["id"], "classified_id": receipt["id"],
        "protocol": receiver.channel_protocol(), "evidence": list(evidence), "selected": chosen,
        "carried_files": {p: descriptor(carried.get(p)) for p in sorted(paths)},
        "included_sources": {p: blob(carried.get(p)) for p in sorted(include_sources)}})


def _valid_descriptor(value: dict | None) -> None:
    if value is not None and (not isinstance(value, dict) or
            set(value) != {"sha256", "size"} or not isinstance(value["sha256"], str) or
            re.fullmatch(r"[a-f0-9]{64}", value["sha256"]) is None or
            type(value["size"]) is not int or value["size"] < 0):
        raise ChannelError("invalid carried raw-file descriptor")


def verify_feedback(manifest: dict, feedback: dict, receiver: ModuleType, *,
                    expected_release_id: str, carried: dict | None = None,
                    receipt: dict | None = None, expected_feedback_id: str | None = None) -> dict:
    """Separate integrity, source-pair membership, receipt linkage and authorship.

    Without source bytes, pair identity is explicitly not recomputed. A seal
    cannot authenticate a reviewer, evidence reference or classification.
    """
    _, after = require_release(manifest, expected_release_id)
    _agree_receiver(manifest, receiver)
    checked(feedback, FEEDBACK_SCHEMA)
    if set(feedback) != {"schema", "release_id", "review_id", "classified_id", "protocol",
                         "evidence", "selected", "carried_files", "included_sources", "id"}:
        raise ChannelError("unexpected feedback fields")
    if feedback["release_id"] != manifest["id"] or feedback["protocol"] != receiver.channel_protocol():
        raise ChannelError("feedback release or normalization authority differs")
    for name in ("review_id", "classified_id"):
        if not isinstance(feedback[name], str) or re.fullmatch(r"[0-9a-f]{64}", feedback[name]) is None:
            raise ChannelError("feedback needs valid review and classification receipt ids")
    if expected_feedback_id is not None and feedback["id"] != expected_feedback_id:
        raise ChannelError("feedback differs from the independently trusted feedback id")
    if (not isinstance(feedback["evidence"], list) or not feedback["evidence"] or
            any(not isinstance(v, str) or not v.strip() for v in feedback["evidence"])):
        raise ChannelError("feedback needs behavioral evidence references")
    sources, identities = feedback["included_sources"], feedback["carried_files"]
    if not isinstance(sources, dict) or not isinstance(identities, dict) or not set(sources) <= set(identities):
        raise ChannelError("source context exceeds selected files")
    raw_sources = {}
    for rel, ref in identities.items():
        owner(manifest["scope"], rel, "carried")
        _valid_descriptor(ref)
        if rel in sources:
            raw_sources[rel] = unblob(sources[rel])
        if carried is not None and rel in carried:
            if rel in raw_sources and raw_sources[rel] != carried[rel]:
                raise ChannelError("included and independently supplied carried bytes disagree")
            raw_sources[rel] = carried[rel]
        if rel in raw_sources and descriptor(raw_sources[rel]) != ref:
            raise ChannelError(f"carried raw bytes differ from the feedback snapshot: {rel}")
    if receipt is not None:
        check_classified(receipt, receiver)
        if receipt["id"] != feedback["classified_id"] or receipt["review"]["id"] != feedback["review_id"]:
            raise ChannelError("feedback names a different classification receipt")
        if receipt["review"]["release_id"] != manifest["id"]:
            raise ChannelError("classification receipt belongs to another release")
    chosen = feedback["selected"]
    if not isinstance(chosen, list) or not chosen:
        raise ChannelError("feedback must select at least one offer")
    seen, indexes, paths = set(), set(), set()
    cache = {}
    verified = 0
    for item in chosen:
        if not isinstance(item, dict) or set(item) != {"review_index", "file_ordinal", "row"}:
            raise ChannelError("feedback selection needs an index, file ordinal and row")
        row, index, ordinal = item["row"], item["review_index"], item["file_ordinal"]
        if (type(index) is not int or index < 0 or index in indexes or
                type(ordinal) is not int or ordinal < 0):
            raise ChannelError("invalid or duplicate feedback index/ordinal")
        if not isinstance(row, dict) or set(row) != {*ROW_FIELDS, *DECISION_FIELDS}:
            raise ChannelError("unexpected measured row fields")
        if any(not isinstance(row[f], str) or not row[f] for f in ROW_FIELDS):
            raise ChannelError("invalid measured row fields")
        for field in ("engine_sha256", "carried_sha256"):
            if re.fullmatch(r"[0-9a-f]{64}", row[field]) is None:
                raise ChannelError("invalid hunk digest")
        _annotation(row, receiver.channel_protocol())
        if row["decision"] != "offer":
            raise ChannelError("feedback row is not a classified offer")
        rel = safe_path(row["file"])
        path = remap(manifest["scope"], rel, "carried")
        engine_rel = str(PurePosixPath(path).relative_to(manifest["scope"]["engine_root"]))
        if row["engine_file"] != engine_rel:
            raise ChannelError("feedback pair does not match the receiver's mapping")
        position = rel, ordinal
        if position in seen:
            raise ChannelError("duplicate feedback occurrence")
        seen.add(position); indexes.add(index); paths.add(rel)
        if receipt is not None:
            recorded = receipt["candidate"]["rows"]
            if (index >= len(recorded) or recorded[index] != row or
                    sum(r["file"] == rel for r in recorded[:index]) != ordinal):
                raise ChannelError("feedback occurrence does not match the supplied review receipt")
        if rel in raw_sources:
            if rel not in cache:
                cache[rel] = receiver.hunks_from_bytes(rel, engine_rel,
                    engine_raw=after.get(path), carried_raw=raw_sources[rel])
            if (ordinal >= len(cache[rel]) or
                    {f: row[f] for f in ROW_FIELDS} != cache[rel][ordinal]):
                raise ChannelError("feedback is not a hunk of the actual engine/carried pair")
            verified += 1
    if paths != set(identities):
        raise ChannelError("carried descriptors must exactly cover selected finding files")
    return {"release_id": manifest["id"], "feedback_id": feedback["id"],
        "integrity": "verified", "release_identity": "matched-independent-id",
        "pair_identities": "recomputed" if verified == len(chosen) else
                           "partially-recomputed" if verified else "not-recomputed",
        "recomputed_occurrences": verified, "selected_occurrences": len(chosen),
        "review_membership": "matched-supplied-receipt" if receipt is not None else "not-checked",
        "feedback_identity": "matched-independent-id" if expected_feedback_id is not None else
                             "not-authenticated",
        "behavioral_evidence": "requires-review", "sources_written": False}


def _write_output(path: Path | None, command: str, payload: dict) -> Path:
    if path is not None:
        write_new(path, payload)
        return path
    import tempfile
    directory = Path("carried-channel-output")
    directory.mkdir(exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=command + "-", suffix=".json", dir=directory)
    with os.fdopen(fd, "w", encoding="ascii", newline="\n") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n")
    return Path(name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    scope = commands.add_parser("scope", help="export the trusted receiver's typed mapping")
    scope.add_argument("--receiver", type=Path, required=True)
    emit = commands.add_parser("emit", help="capture scoped immutable old/new Git snapshots")
    for flag in ("repo", "scope"):
        emit.add_argument("--" + flag, type=Path, required=True)
    emit.add_argument("--old-repo", type=Path, help="read the old endpoint from a separate public mirror")
    for flag in ("old", "new", "old-version", "new-version"):
        emit.add_argument("--" + flag, required=True)
    check = commands.add_parser("verify-git", help="check named Git objects, not publisher approval")
    check.add_argument("--manifest", type=Path, required=True)
    check.add_argument("--repo", type=Path, required=True)
    check.add_argument("--old-repo", type=Path)
    review = commands.add_parser("review", help="measure the actual pairs; never merge physics")
    classify = commands.add_parser("classify", help="retain authored classifications and retirement acknowledgements")
    candidate = commands.add_parser("candidate", help="export a newly validated legacy registry candidate")
    feedback = commands.add_parser("feedback", help="offer selected rows, metadata only by default")
    verify = commands.add_parser("verify-feedback", help="report exactly which identities were verified")
    for cmd in (scope, emit, review, classify, candidate, feedback):
        cmd.add_argument("--out", type=Path, help="new output path; otherwise allocate a new owned generation")
    for cmd in (review, candidate, feedback, verify):
        cmd.add_argument("--manifest", type=Path, required=True)
        cmd.add_argument("--receiver", type=Path, required=True)
        cmd.add_argument("--expected-release-id", required=True,
                         help="independent trusted id, not an id copied from received JSON")
    for cmd in (review, candidate, feedback):
        cmd.add_argument("--carried", type=Path, required=True)
    prior = review.add_mutually_exclusive_group(required=True)
    prior.add_argument("--previous", type=Path)
    prior.add_argument("--bootstrap", action="store_true", help="no inherited classifications")
    for flag in ("review", "decisions", "receiver"):
        classify.add_argument("--" + flag, type=Path, required=True)
    classify.add_argument("--expected-review-id", required=True)
    for cmd in (candidate, feedback):
        cmd.add_argument("--classified", type=Path, required=True)
    feedback.add_argument("--index", type=int, action="append", required=True)
    feedback.add_argument("--evidence", action="append", required=True)
    feedback.add_argument("--include-source", action="append", default=[],
                          help="explicit selected carried path whose whole bytes may be exported")
    verify.add_argument("--feedback", type=Path, required=True)
    verify.add_argument("--carried", type=Path)
    verify.add_argument("--classified", type=Path)
    verify.add_argument("--expected-feedback-id")
    verify.add_argument("--require-pairs", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "scope":
            result = load_receiver(args.receiver).channel_scope()
            validate_scope(result)
        elif args.command == "emit":
            result = create_release(args.repo, args.old, args.new, read_json(args.scope),
                                    args.old_version, args.new_version, old_repo=args.old_repo)
        elif args.command == "verify-git":
            verify_git(read_json(args.manifest), args.repo, old_repo=args.old_repo)
            print("scoped bytes and directory coverage match the named Git objects; publisher not authenticated")
            return 0
        elif args.command == "classify":
            result = classify_review(read_json(args.review), read_json(args.decisions),
                                     load_receiver(args.receiver), expected_review_id=args.expected_review_id)
        else:
            manifest = read_json(args.manifest)
            receiver = load_receiver(args.receiver)
            carried = read_carried(receiver.channel_scope(), args.carried) if args.carried is not None else None
            if args.command == "review":
                result = review_release(manifest, receiver, carried,
                    None if args.bootstrap else read_json(args.previous),
                    expected_release_id=args.expected_release_id)
            elif args.command == "candidate":
                result = candidate_document(manifest, receiver, carried, read_json(args.classified),
                                            expected_release_id=args.expected_release_id)
            elif args.command == "feedback":
                result = make_feedback(manifest, receiver, carried, read_json(args.classified),
                    args.index, args.evidence, expected_release_id=args.expected_release_id,
                    include_sources=args.include_source)
            else:
                result = verify_feedback(manifest, read_json(args.feedback), receiver,
                    expected_release_id=args.expected_release_id, carried=carried,
                    receipt=None if args.classified is None else read_json(args.classified),
                    expected_feedback_id=args.expected_feedback_id)
                print(json.dumps(result, indent=2))
                if args.require_pairs and result["pair_identities"] != "recomputed":
                    raise ChannelError("full pair verification requires every selected carried file's exact bytes")
                return 0
        path = _write_output(args.out, args.command, result)
        print(json.dumps({"output": str(path), "id": result.get("id"), "command": args.command}))
        return 1 if args.command == "review" and result["needs_review"] else 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"carried-physics channel: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
