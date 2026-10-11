"""Reader for the fast SBM Fortran oracle fixtures (tests/data/fast_sbm_oracle).

This module is the only reader of those files. Every fixture is a gzip file
holding one record stream exactly as the oracle harness wrote it on the
build node (``fsbm_probe.F90``, module ``fsbm_io``): a sequence of records,
each an int32 type tag followed by that type's fields, raw little-endian, in
Fortran order. ``MANIFEST.json`` beside the fixtures carries, per file, the
record layout (types, field names, kinds and shapes), the record counts, the
uncompressed byte count and the SHA-256 of the uncompressed bytes. Nothing is
returned before the bytes match that hash.

Arrays keep Fortran index order: a field the harness declared ``(33, 40)``
comes back with shape ``(33, 40)`` and ``a[i - 1, j - 1]`` is ``A(I, J)``.
Fields written as ``f4`` (the native WRF build, default real) stay float32,
``f8`` fields are float64, ``i4`` fields int32; nothing is converted, and a
scalar field comes back as a numpy scalar of that dtype.

Public entry points (numpy only; no CUDA, no Fortran, no network)::

    load_manifest(root=None) -> dict
    fixture_names(root=None) -> list[str]
    load_fixture(name, root=None) -> dict[str, numpy.ndarray]
        keys "<record type>.<field>", each stacked over that type's records
        with a leading record axis (shape (count, *field shape))
    iter_records(name, root=None, types=None) -> iterator of Record
        records in stream order (probe streams interleave record types in the
        order WRF reached the anchors)
    iter_replay(name, root=None) -> iterator of (step, pre, post)
        the kc1_replay_*, kc2_replay_* and kc2g_replay_* families: the
        pre-call and post-call state of each replay step; the crossreplay
        families yield (step, None, post)
    iter_probe(name, root=None, step=None, types=None) -> iterator of Record
        the kc1_probe_* and kc2_probe_* families, optionally one step or some
        record types
    iter_refusals(root=None, build=None, name="refusals") -> iterator of Refusal
        a refusals container ("refusals", the warm cases of milestone 1, or
        "refusals_cold", the cold cases of milestone 2): each case's input
        records, the message wrf_error_fatal wrote and the process exit status

``parse_layout`` and ``parse_stream`` also read the raw ``.layout``/``.raw``
pairs the harness writes, which is how ``pack_fixtures.py`` validates what it
packs; they carry no hash check of their own.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "tests" / "data" / "fast_sbm_oracle"
MANIFEST_NAME = "MANIFEST.json"
LAYOUT_MAGIC = "fsbm-layout 1"
KIND_DTYPES = {"f4": "<f4", "f8": "<f8", "i4": "<i4"}


class FixtureError(RuntimeError):
    """A fixture is missing, does not match its manifest, or is malformed."""


@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: str                  # "f4", "f8" or "i4"
    shape: tuple[int, ...]     # Fortran dimensions, () for a scalar

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(KIND_DTYPES[self.kind])

    @property
    def nbytes(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n * self.dtype.itemsize


@dataclass(frozen=True)
class RecordType:
    tag: int
    name: str
    fields: tuple[FieldSpec, ...]
    count: int

    @property
    def nbytes(self) -> int:
        return 4 + sum(f.nbytes for f in self.fields)


@dataclass(frozen=True)
class Layout:
    types: tuple[RecordType, ...]

    def by_tag(self) -> dict[int, RecordType]:
        return {t.tag: t for t in self.types}

    def by_name(self) -> dict[str, RecordType]:
        return {t.name: t for t in self.types}

    @property
    def nbytes(self) -> int:
        return sum(t.count * t.nbytes for t in self.types)

    def to_json(self) -> list[dict]:
        """Manifest form: each field is "<name> <kind> <Fortran dims...>"."""

        return [
            {
                "tag": t.tag,
                "name": t.name,
                "count": t.count,
                "record_bytes": t.nbytes,
                "fields": [" ".join([f.name, f.kind, *map(str, f.shape)])
                           for f in t.fields],
            }
            for t in self.types
        ]

    @classmethod
    def from_json(cls, items: list[dict]) -> "Layout":
        types = []
        for item in items:
            fields = []
            for text in item["fields"]:
                parts = text.split()
                if len(parts) < 2 or parts[1] not in KIND_DTYPES:
                    raise FixtureError(f"bad field entry {text!r}")
                fields.append(FieldSpec(parts[0], parts[1],
                                        tuple(int(d) for d in parts[2:])))
            rtype = RecordType(int(item["tag"]), item["name"], tuple(fields),
                               int(item["count"]))
            if "record_bytes" in item and int(item["record_bytes"]) != rtype.nbytes:
                raise FixtureError(f"type {rtype.name}: record_bytes disagrees with its fields")
            types.append(rtype)
        return cls(tuple(types))


@dataclass(frozen=True)
class Record:
    type: str
    index: int                 # position of this record among its type
    fields: dict[str, np.ndarray | int | float] = field(repr=False)

    def __getitem__(self, key: str):
        return self.fields[key]


@dataclass(frozen=True)
class Refusal:
    case: int
    build: str
    routine: str
    exit_status: int
    message: str
    records: tuple[Record, ...]


# --------------------------------------------------------------------- layout
def parse_layout(text: str) -> Layout:
    """Parse a harness ``.layout`` file."""

    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if not lines or lines[0] != LAYOUT_MAGIC:
        raise FixtureError("layout does not start with the fsbm-layout header")
    if lines[-1] != "end":
        raise FixtureError("layout is not closed (harness stopped mid-stream)")
    types: dict[int, dict] = {}
    order: list[int] = []
    current = None
    counts: dict[int, int] = {}
    for ln in lines[1:-1]:
        parts = ln.split()
        if parts[0] == "type":
            tag, name, nf = int(parts[1]), parts[2], int(parts[3])
            if tag in types:
                raise FixtureError(f"record type {tag} defined twice")
            types[tag] = {"name": name, "nfields": nf, "fields": []}
            order.append(tag)
            current = tag
        elif parts[0] == "field":
            if current is None:
                raise FixtureError("field line before any type line")
            name, kind, rank = parts[1], parts[2], int(parts[3])
            if kind not in KIND_DTYPES:
                raise FixtureError(f"unknown field kind {kind!r}")
            shape = tuple(int(d) for d in parts[4:4 + rank])
            if len(shape) != rank:
                raise FixtureError(f"field {name}: rank {rank} with {len(shape)} dims")
            types[current]["fields"].append(FieldSpec(name, kind, shape))
        elif parts[0] == "count":
            counts[int(parts[1])] = int(parts[2])
        else:
            raise FixtureError(f"unexpected layout line {ln!r}")
    out = []
    for tag in order:
        t = types[tag]
        if len(t["fields"]) != t["nfields"]:
            raise FixtureError(f"type {t['name']}: field count mismatch")
        if tag not in counts:
            raise FixtureError(f"type {t['name']}: no count line")
        out.append(RecordType(tag, t["name"], tuple(t["fields"]), counts[tag]))
    return Layout(tuple(out))


def _field_value(buf: memoryview, offset: int, spec: FieldSpec):
    n = spec.nbytes
    arr = np.frombuffer(buf[offset:offset + n], dtype=spec.dtype)
    if spec.shape:
        value = arr.reshape(spec.shape, order="F").copy()
    else:
        value = arr[0]          # a numpy scalar of the field's own dtype
    return value, offset + n


def parse_stream(data: bytes, layout: Layout,
                 types: Iterable[str] | None = None) -> Iterator[Record]:
    """Yield the records of a stream in order, checking it against ``layout``."""

    wanted = None if types is None else set(types)
    by_tag = layout.by_tag()
    seen = {t.tag: 0 for t in layout.types}
    buf = memoryview(data)
    offset = 0
    total = len(data)
    while offset < total:
        if offset + 4 > total:
            raise FixtureError("stream ends inside a record tag")
        tag = int(np.frombuffer(buf[offset:offset + 4], dtype="<i4")[0])
        offset += 4
        rtype = by_tag.get(tag)
        if rtype is None:
            raise FixtureError(f"record tag {tag} is not in the layout")
        if offset + rtype.nbytes - 4 > total:
            raise FixtureError(f"stream ends inside a {rtype.name} record")
        index = seen[tag]
        seen[tag] += 1
        if wanted is not None and rtype.name not in wanted:
            offset += rtype.nbytes - 4
            continue
        values = {}
        for spec in rtype.fields:
            values[spec.name], offset = _field_value(buf, offset, spec)
        yield Record(rtype.name, index, values)
    for t in layout.types:
        if seen[t.tag] != t.count:
            raise FixtureError(
                f"type {t.name}: {seen[t.tag]} records in the stream, "
                f"layout says {t.count}")


# ------------------------------------------------------------------ manifest
def _root(root: str | Path | None) -> Path:
    return FIXTURE_ROOT if root is None else Path(root)


def load_manifest(root: str | Path | None = None) -> dict:
    path = _root(root) / MANIFEST_NAME
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FixtureError(f"no oracle manifest at {path}") from exc


def fixture_names(root: str | Path | None = None) -> list[str]:
    return sorted(load_manifest(root)["files"])


def _entry(name: str, manifest: dict) -> dict:
    try:
        return manifest["files"][name]
    except KeyError as exc:
        raise FixtureError(f"no fixture named {name!r} in the manifest") from exc


def read_fixture_bytes(name: str, root: str | Path | None = None,
                       manifest: dict | None = None) -> bytes:
    """Return a fixture's uncompressed bytes after the SHA-256 check."""

    manifest = load_manifest(root) if manifest is None else manifest
    entry = _entry(name, manifest)
    path = _root(root) / entry["file"]
    try:
        raw = gzip.decompress(path.read_bytes())
    except FileNotFoundError as exc:
        raise FixtureError(f"fixture file {path} is missing") from exc
    except (OSError, EOFError, zlib.error) as exc:
        raise FixtureError(f"fixture file {path} is not a valid gzip stream: {exc}") from exc
    if len(raw) != entry["bytes"]:
        raise FixtureError(
            f"{name}: {len(raw)} uncompressed bytes, manifest says {entry['bytes']}")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != entry["sha256"]:
        raise FixtureError(
            f"{name}: SHA-256 {digest} does not match the manifest {entry['sha256']}")
    return raw


def _layout(entry: dict) -> Layout:
    if "layout" not in entry:
        raise FixtureError("this fixture is a container; use iter_refusals")
    return Layout.from_json(entry["layout"])


# ------------------------------------------------------------------- readers
def iter_records(name: str, root: str | Path | None = None,
                 types: Iterable[str] | None = None) -> Iterator[Record]:
    manifest = load_manifest(root)
    entry = _entry(name, manifest)
    layout = _layout(entry)
    data = read_fixture_bytes(name, root, manifest)
    yield from parse_stream(data, layout, types)


def load_fixture(name: str, root: str | Path | None = None) -> dict[str, np.ndarray]:
    """All records of a fixture as named arrays, ``"<type>.<field>"``."""

    manifest = load_manifest(root)
    entry = _entry(name, manifest)
    if entry.get("container") == "refusals":
        return _load_refusals_arrays(root, manifest, name)
    layout = _layout(entry)
    data = read_fixture_bytes(name, root, manifest)
    collected: dict[str, list] = {}
    for rec in parse_stream(data, layout):
        for key, value in rec.fields.items():
            collected.setdefault(f"{rec.type}.{key}", []).append(value)
    out: dict[str, np.ndarray] = {}
    for t in layout.types:
        for spec in t.fields:
            key = f"{t.name}.{spec.name}"
            values = collected.get(key, [])
            if spec.shape:
                arr = (np.stack(values) if values
                       else np.zeros((0, *spec.shape), dtype=spec.dtype))
            else:
                arr = np.asarray(values, dtype=spec.dtype)
            out[key] = arr
    return out


def iter_replay(name: str, root: str | Path | None = None):
    """(step, pre, post) for kc1_replay_*; (step, None, post) for cross-replay."""

    pending = None
    for rec in iter_records(name, root):
        if rec.type == "pre":
            if pending is not None:
                raise FixtureError(f"{name}: two pre records without a post")
            pending = rec
        elif rec.type == "post":
            step = int(rec["step"])
            if pending is not None and int(pending["step"]) != step:
                raise FixtureError(f"{name}: pre step {pending['step']} then post {step}")
            yield step, pending, rec
            pending = None
    if pending is not None:
        raise FixtureError(f"{name}: stream ends after a pre record")


def iter_probe(name: str, root: str | Path | None = None, step: int | None = None,
               types: Iterable[str] | None = None) -> Iterator[Record]:
    for rec in iter_records(name, root, types):
        if step is not None and "step" in rec.fields and int(rec["step"]) != step:
            continue
        yield rec


def iter_refusals(root: str | Path | None = None, build: str | None = None,
                  name: str = "refusals") -> Iterator[Refusal]:
    manifest = load_manifest(root)
    entry = _entry(name, manifest)
    if entry.get("container") != "refusals":
        raise FixtureError(f"the {name} fixture is not a refusals container")
    data = read_fixture_bytes(name, root, manifest)
    for member in entry["members"]:
        if build is not None and member["build"] != build:
            continue
        start, length = member["offset"], member["bytes"]
        chunk = data[start:start + length]
        if len(chunk) != length:
            raise FixtureError("refusals container is shorter than its members")
        if hashlib.sha256(chunk).hexdigest() != member["sha256"]:
            raise FixtureError(f"refusal case {member['case']} ({member['build']}) "
                               "does not match its member hash")
        layout = Layout.from_json(member["layout"])
        records = tuple(parse_stream(chunk, layout))
        yield Refusal(member["case"], member["build"], member["routine"],
                      member["exit_status"], member["message"], records)


def _load_refusals_arrays(root, manifest, name="refusals") -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for ref in iter_refusals(root, name=name):
        prefix = f"case{ref.case}_{ref.build}"
        out[f"{prefix}.exit_status"] = np.asarray(ref.exit_status, dtype="<i4")
        for rec in ref.records:
            for key, value in rec.fields.items():
                out[f"{prefix}.{rec.type}.{key}"] = np.asarray(value)
    return out


__all__ = [
    "FIXTURE_ROOT", "FieldSpec", "FixtureError", "Layout", "Record", "RecordType",
    "Refusal", "fixture_names", "iter_probe", "iter_records", "iter_refusals",
    "iter_replay", "load_fixture", "load_manifest", "parse_layout", "parse_stream",
    "read_fixture_bytes",
]
