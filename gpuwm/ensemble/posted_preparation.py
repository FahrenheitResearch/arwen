"""Append-only physical member authority at ordinary native preparation seams.

The provider owns physical mapping and recentering. This module verifies its
native bytes against the actual preparer's source, grid, statics and field
contract, then returns the snapshot the unchanged initializer consumes.
An inactive scope returns the original object and publishes nothing.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType
import uuid

import numpy as np


HEAD_SCHEMA = "gpuwm-ensemble-posted-member-head.v1"
FRAME_SCHEMA = "gpuwm-ensemble-posted-member-frame.v1"
SEGMENT_SCHEMA = "gpuwm-ensemble-posted-member-segment.v1"
SEAL_SCHEMA = "gpuwm-ensemble-posted-member-seal.v1"
BINDING_KEY = "ensemble_posted_member_input"
_CURRENT = ContextVar("gpuwm_posted_member_preparation", default=None)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _frozen(value):
    return json.loads(_canonical(value))


def _utc(value, *, native=False):
    if not isinstance(value, datetime):
        raise ValueError("posted member preparation needs an actual valid-time datetime")
    if value.tzinfo is None:
        if not native:
            raise ValueError("posted member plans need explicit UTC offsets")
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _publish(path, document):
    """Publish once; an existing byte-identical authority is idempotent."""
    path = Path(path)
    payload = (_canonical(document) + "\n").encode()
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("a posted member authority changed after publication")
        return hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".partial")
    try:
        with temporary.open("xb") as writer:
            writer.write(payload)
            writer.flush()
            os.fsync(writer.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def _trajectory(value):
    return {"source": value.source, "cycle": value.cycle.isoformat(),
            "member": value.member, "identity": value.identity}


def _snapshot_words(snapshot):
    """Provenance only: existing transfer, native dtype, exact stored words."""
    from gpuwm.ingest.preprocess_backend import _host
    arrays = {"field__" + key: value for key, value in snapshot.fields.items()}
    arrays["levels_hpa"] = snapshot.levels_hpa
    for key, value in vars(snapshot).items():
        if key != "fields" and key != "levels_hpa" and hasattr(value, "dtype") and hasattr(value, "shape"):
            arrays["meta__" + key] = value
    result = {}
    for name, value in sorted(arrays.items()):
        host = _host(value)
        if host.dtype.hasobject:
            raise ValueError("native physical provenance cannot contain object arrays")
        result[name] = {"dtype": host.dtype.str, "shape": list(host.shape),
                        "sha256": hashlib.sha256(host.tobytes(order="C")).hexdigest()}
    def metadata(value):
        if isinstance(value, Mapping):
            return {str(key): metadata(item) for key, item in sorted(value.items())}
        if isinstance(value, (tuple, list)):
            return [metadata(item) for item in value]
        if value is None or isinstance(value, str):
            return value
        if isinstance(value, (bool, int, float, np.generic)):
            array = np.asarray(value)
            return {"dtype": array.dtype.str, "words": array.tobytes().hex()}
        raise TypeError("native physical snapshot contains unsupported scalar metadata")
    result["metadata"] = {key: metadata(value) for key, value in vars(snapshot).items()
        if key not in ("fields", "levels_hpa", "valid_time") and not hasattr(value, "dtype")}
    result["valid_time"] = _utc(snapshot.valid_time, native=True).isoformat()
    return result


@dataclass(frozen=True)
class NativeSnapshotContext:
    snapshot: object
    grid: object
    cfg: object
    static_fields: object
    landuse_attrs: object
    metadata: object
    domain_id: int


@dataclass(frozen=True)
class NativeSnapshotAuthority:
    """Source-adapter metadata resolved from the actual native call.

    The resolver must use the ordinary preparer's verified source receipts
    and qualified native field contract. Units are never inferred here.
    """
    source_identity: dict
    field_contract: dict
    input_plan_sha256: str


@dataclass(frozen=True)
class PostedPreparedMemberInput:
    """An initial head plus independently bound later native forcing knots."""
    member_id: int
    seed: int
    recipe_sha256: str
    trajectory: dict
    geometry_sha256: str
    valid_times: tuple[datetime, ...]
    head: dict
    head_sha256: str

    def receipt(self):
        return {"schema": HEAD_SCHEMA, "head_sha256": self.head_sha256,
                "head": _frozen(self.head)}


class PostedMemberPreparation:
    """One member/domain, using the provider's original global member index.

    ``authority_resolver`` is supplied by registered source-preparation
    metadata and receives the actual native call. The provider's documented
    ``prepare`` and stream ``require`` APIs remain responsible for waiting
    and numeric operations. No full future manifest is required at startup.
    """
    def __init__(self, provider, *, member_id, valid_times, output_root,
                 authority_resolver, domain_id=1, _reader=None,
                 _geometry=None, _statics=None, _words=None):
        self.provider, self.recipe = provider, provider.recipe
        by_id = {member.index: member for member in self.recipe.members}
        if type(member_id) is not int or member_id not in by_id:
            raise ValueError("posted preparation selected an index outside the original recipe")
        self.member = by_id[member_id]
        self.member_id, self.domain_id = int(member_id), int(domain_id)
        self.times = tuple(_utc(value) for value in valid_times)
        if (not self.times or self.times[0] != self.recipe.start or self.times[-1] != self.recipe.end
                or any(a >= b for a, b in zip(self.times, self.times[1:]))):
            raise ValueError("posted member knots must cover the exact recipe window in increasing order")
        self.root = Path(output_root).resolve()
        if self.root.exists():
            raise FileExistsError("posted member preparation requires a new owned directory")
        self.root.mkdir(parents=True)
        self.authority_resolver = authority_resolver
        self._reader, self._geometry, self._statics = _reader, _geometry, _statics
        if any(value is not None for value in (_reader, _geometry, _statics)) and not all(
                callable(value) for value in (_reader, _geometry, _statics)):
            raise TypeError("native test adapters must provide the reader, geometry and static verifiers together")
        self._words = _words or _snapshot_words
        self.owner = None
        self.frames, self._authorities = {}, {}
        self._sealed = False

    def _native_functions(self):
        if self._reader is None:
            from gpuwm.ensemble.physical_store import NativePhysicalStore, physical_static_identity
            from gpuwm.native_wrf_contract import native_geometry_contract, native_static_export_fields
            self._reader = NativePhysicalStore
            self._geometry = native_geometry_contract
            self._statics = lambda fields, grid, attrs: physical_static_identity(
                native_static_export_fields(fields, grid), attrs)
        return self._reader, self._geometry, self._statics

    def _ordinary_stream(self):
        source = self.recipe.base if self.recipe.kind == "recentered" else self.member.trajectory
        return self.provider.streams[source.identity]

    def _check_head(self):
        if self.owner is not None and (
                _sha(self.root / "member-head.json") != self.owner.head_sha256
                or hashlib.sha256((_canonical(self.owner.head) + "\n").encode()).hexdigest()
                != self.owner.head_sha256):
            raise ValueError("posted member head changed after its native authority was loaded")

    def replace_native(self, context):
        """Verify and load the actual member bytes before native initialization."""
        self._check_head()
        if not isinstance(context, NativeSnapshotContext) or int(context.domain_id) != self.domain_id:
            raise ValueError("posted native call belongs to a different domain owner")
        instant = _utc(context.snapshot.valid_time, native=True)
        if instant not in self.times:
            raise ValueError("the native initializer requested a knot outside the member plan")
        index = self.times.index(instant)
        authority = self.authority_resolver(context)
        if not isinstance(authority, NativeSnapshotAuthority):
            raise TypeError("source preparation metadata must return NativeSnapshotAuthority")
        reader, geometry_reader, static_reader = self._native_functions()
        geometry = _frozen(geometry_reader(context.grid, context.cfg))
        statics = _frozen(static_reader(context.static_fields, context.grid, context.landuse_attrs))
        stream = self._ordinary_stream()
        source = {**_frozen(authority.source_identity), "static_identity": statics}
        if (stream.head["grid"] != geometry or stream.head["source"] != source
                or stream.head["field_contract"] != authority.field_contract
                or stream.head["input_plan_sha256"] != authority.input_plan_sha256):
            raise ValueError("posted physical source differs from the actual native source/grid/static/units authority")
        base_store, _ = stream.require(instant)
        if self._words(base_store.read(0)) != self._words(context.snapshot):
            raise ValueError("posted base physical bytes differ from the ordinary native mapping")
        native = {"grid": geometry, "static": statics,
                  "source": _frozen(authority.source_identity),
                  "field_contract": _frozen(authority.field_contract),
                  "input_plan_sha256": authority.input_plan_sha256}
        if self.owner is None:
            head = {"schema": HEAD_SCHEMA, "member_id": self.member_id, "seed": self.member.seed,
                    "domain_id": self.domain_id, "recipe_sha256": self.recipe.sha256,
                    "trajectory": _trajectory(self.member.trajectory),
                    "ordinary_trajectory": _trajectory(stream.trajectory),
                    "valid_times": [value.isoformat() for value in self.times], "native": native,
                    "required_sources": sorted(({self.recipe.base.identity,
                        *(value.identity for value in self.recipe.donor_population)}
                        if self.recipe.kind == "recentered" else {self.member.trajectory.identity})),
                    "source_heads": {key: value.head_sha256 for key, value in sorted(self.provider.streams.items())}}
            sha = _publish(self.root / "member-head.json", head)
            self.owner = PostedPreparedMemberInput(self.member_id, self.member.seed, self.recipe.sha256,
                _trajectory(self.member.trajectory), _digest(geometry), self.times, head, sha)
        elif self.owner.head["native"] != native:
            raise ValueError("posted member native authority changed between forcing knots")
        if index not in self.frames:
            if self._sealed:
                raise ValueError("a sealed member cannot publish another native knot")
            output = self.root / "physical" / f"{index:05d}"
            values = self.provider.prepare(instant, {self.member_id: output},
                work_root=self.root / "work" / (f"{index:05d}-" + uuid.uuid4().hex))
            item = values[self.member_id]
            store = reader(output)
            snapshot = store.read(0)
            sidecar = output / "posted-physical-receipt.json"
            receipt = json.loads(sidecar.read_text())
            if receipt != item["receipt"]:
                raise ValueError("posted provider sidecar differs from its returned authority")
            if (receipt.get("member_index") != self.member_id or receipt.get("member_seed") != self.member.seed
                    or receipt.get("recipe_sha256") != self.recipe.sha256
                    or receipt.get("trajectory") != _trajectory(self.member.trajectory)
                    or receipt.get("valid_time") != instant.isoformat()
                    or receipt.get("store_sha256") != store.manifest_sha256
                    or item.get("sha256") != store.manifest_sha256
                    or Path(item.get("path", "")).resolve() != store.manifest_path.resolve()
                    or store.document["grid"] != geometry or store.field_contract != authority.field_contract
                    or tuple(_utc(time, native=True) for time in store.times) != (instant,)):
                raise ValueError("posted member bytes lost their original recipe/source/grid/units identity")
            prepared_source = store.document.get("source")
            if self.recipe.kind == "recentered":
                population = {value.identity for value in self.recipe.donor_population}
                if (not isinstance(prepared_source, dict)
                        or prepared_source.get("schema") != "gpuwm-ensemble-recentered-preparation.v1"
                        or prepared_source.get("selected_member") != self.member.trajectory.identity
                        or prepared_source.get("base", {}).get("source") != source
                        or set(prepared_source.get("donors", {})) != population):
                    raise ValueError("member physical store lost its base or complete recentering operator authority")
            elif prepared_source != source:
                raise ValueError("direct member physical store belongs to another ordinary native source")
            expected_sources = ({self.recipe.base.identity, *(value.identity for value in self.recipe.donor_population)}
                                if self.recipe.kind == "recentered" else {self.member.trajectory.identity})
            if set(receipt.get("sources", {})) != expected_sources:
                raise ValueError("posted member receipt omitted a required frozen source trajectory")
            record = {"schema": FRAME_SCHEMA, "head_sha256": self.owner.head_sha256,
                      "index": index, "valid_time": instant.isoformat(),
                      "store": {"manifest": _frozen(store.document), "sha256": store.manifest_sha256},
                      "provider_receipt": _frozen(receipt), "provider_receipt_sha256": _sha(sidecar),
                      "native_words_sha256": _digest(self._words(snapshot))}
            self._check_consumed_sources(record)
            self.frames[index] = {"record": record,
                "sha256": _publish(self.root / "ready" / f"{index:05d}.json", record), "store": output}
            self._authorities[index] = _frozen(authority.source_identity)
            return snapshot
        record = self.frames[index]
        if _sha(self.root / "ready" / f"{index:05d}.json") != record["sha256"]:
            raise ValueError("posted member ready authority changed after consumption")
        self._check_consumed_sources(record["record"])
        return reader(record["store"]).read(0)

    def _check_consumed_sources(self, record):
        self._check_head()
        instant = datetime.fromisoformat(record["valid_time"])
        for key, rows in record["provider_receipt"]["sources"].items():
            stream = self.provider.streams[key]
            expected = ((instant,) if self.recipe.kind != "recentered" or key == self.recipe.base.identity
                        else tuple(stream.bracket_times(instant)))
            if tuple(datetime.fromisoformat(row["marker"]["valid_time"]) for row in rows) != expected:
                raise ValueError("posted member source receipts do not bind its exact planned time brackets")
            for row in rows:
                _, actual = stream.require(datetime.fromisoformat(row["marker"]["valid_time"]))
                if actual != row or row["head_sha256"] != self.owner.head["source_heads"][key]:
                    raise ValueError("a consumed posted source head or marker changed")

    def bind_head(self):
        self._check_head()
        if self.owner is None or 0 not in self.frames:
            raise ValueError("a prepared member head requires its actual initialized physical frame")
        return {**self.owner.receipt(), "initial": _frozen(self.frames[0]["record"]),
                "initial_sha256": self.frames[0]["sha256"]}

    def verify_prepared_source(self, identity):
        """Hold the portable prepared cache to the source used at real init."""
        self._check_head()
        source = identity.get("source_identity", {})
        if "root_preparation" in source:
            source = source["root_preparation"]
        if not isinstance(source, Mapping) or not _source_matches(self._authorities[0], source,
                self.owner.head["native"]["input_plan_sha256"]):
            raise ValueError("prepared member head differs from the ordinary native source initialized")

    def bind_segment(self, index):
        self._check_head()
        if type(index) is not int or not 0 <= index < len(self.times) - 1:
            raise ValueError("posted member segment is outside its planned forcing window")
        if index not in self.frames or index + 1 not in self.frames:
            raise ValueError("a boundary segment requires both actual member forcing knots")
        frames = [self.frames[position] for position in (index, index + 1)]
        for frame in frames:
            self._check_consumed_sources(frame["record"])
        return {"schema": SEGMENT_SCHEMA, "head_sha256": self.owner.head_sha256,
                "index": index, "frames": [_frozen(value["record"]) for value in frames],
                "frame_sha256": [value["sha256"] for value in frames]}

    def seal(self):
        self._check_head()
        if set(self.frames) != set(range(len(self.times))):
            raise ValueError("posted member preparation cannot seal missing native forcing knots")
        for frame in self.frames.values():
            self._check_consumed_sources(frame["record"])
        document = {"schema": SEAL_SCHEMA, "head_sha256": self.owner.head_sha256,
                    "frames": [self.frames[index]["sha256"] for index in range(len(self.times))]}
        _publish(self.root / "member-seal.json", document)
        self._sealed = True
        return document

    def receipt(self):
        return {"member_id": self.member_id, "domain_id": self.domain_id,
                "head_sha256": None if self.owner is None else self.owner.head_sha256,
                "ready_knots": sorted(self.frames), "pending_knots": [index for index in range(len(self.times))
                    if index not in self.frames], "sealed": self._sealed,
                "execution": "native per-knot provider; original source waiting and initializer"}


def current_posted_preparation():
    return _CURRENT.get()


def require_exclusive_native_consumer(*, physical_input_store=None,
                                      physical_input_provider=None):
    """Prevent two bound consumers from replacing the same native snapshot."""
    if ((physical_input_store is not None or physical_input_provider is not None)
            and current_posted_preparation() is not None):
        raise ValueError("native physical inputs need one replacement authority; "
                         "a posted scope cannot also select a store or provider")


@contextmanager
def posted_preparation_scope(owner):
    token = _CURRENT.set(owner)
    try:
        yield owner
    finally:
        _CURRENT.reset(token)


def replace_current_native_snapshot(snapshot, *, grid, cfg, static_fields,
                                    landuse_attrs=None, metadata=None, domain_id=1):
    owner = current_posted_preparation()
    if owner is None:
        return snapshot
    if isinstance(owner, dict):
        owner = owner[int(domain_id)]
    return owner.replace_native(NativeSnapshotContext(snapshot, grid, cfg, static_fields,
        landuse_attrs, MappingProxyType({} if metadata is None else metadata), int(domain_id)))


def current_posted_domain(domain_id=1):
    owner = current_posted_preparation()
    if owner is None:
        return None
    if isinstance(owner, dict):
        if int(domain_id) not in owner:
            raise ValueError("posted preparation has no native source owner for this domain")
        return owner[int(domain_id)]
    if int(domain_id) != owner.domain_id:
        raise ValueError("posted preparation uses a different native source domain")
    return owner


def bind_current_prepared_identity(identity, *, domain_id=1):
    owner = current_posted_domain(domain_id)
    if owner is None:
        return identity
    owner.verify_prepared_source(identity)
    result = _frozen(identity)
    target = result["source_identity"]
    if "root_preparation" in target:
        target = target["root_preparation"]
    target[BINDING_KEY] = owner.bind_head()
    return result


def bind_current_source_identity(source_identity, *, domain_id=1):
    owner = current_posted_domain(domain_id)
    if owner is None:
        return source_identity
    owner.verify_prepared_source({"source_identity": source_identity})
    result = _frozen(source_identity)
    result[BINDING_KEY] = owner.bind_head()
    return result


def posted_binding_from_identity(identity):
    source = identity.get("source_identity", {})
    if "root_preparation" in source:
        source = source["root_preparation"]
    return source.get(BINDING_KEY)


def _source_matches(expected, actual, plan):
    from gpuwm.ingest.boundary_stream import as_posted_placeholder
    for key, value in expected.items():
        if actual.get(key) == value:
            continue
        if (key == "input_manifest_sha256" and value == as_posted_placeholder(plan)
                and isinstance(actual.get(key), str) and len(actual[key]) == 64
                and all(character in "0123456789abcdef" for character in actual[key])):
            continue
        return False
    return True


def _published_digest(document):
    return hashlib.sha256((_canonical(document) + "\n").encode()).hexdigest()


def _validate_frame(frame, head, index, digest):
    if (not isinstance(frame, dict) or frame.get("schema") != FRAME_SCHEMA
            or frame.get("head_sha256") != _published_digest(head)
            or frame.get("index") != index or frame.get("valid_time") != head["valid_times"][index]
            or _published_digest(frame) != digest):
        raise ValueError("prepared posted member frame differs from its immutable head/time/digest")
    receipt = frame.get("provider_receipt", {})
    if (receipt.get("member_index") != head["member_id"] or receipt.get("member_seed") != head["seed"]
            or receipt.get("recipe_sha256") != head["recipe_sha256"]
            or receipt.get("trajectory") != head["trajectory"]
            or receipt.get("valid_time") != frame["valid_time"]
            or _published_digest(receipt) != frame.get("provider_receipt_sha256")
            or receipt.get("store_sha256") != frame.get("store", {}).get("sha256")
            or _published_digest(frame["store"]["manifest"]) != frame["store"]["sha256"]
            or frame["store"]["manifest"].get("grid") != head["native"]["grid"]
            or frame["store"]["manifest"].get("field_contract") != head["native"]["field_contract"]
            or set(receipt.get("sources", {})) != set(head["required_sources"])):
        raise ValueError("prepared posted member frame lost its provider/native payload authority")
    for source, rows in receipt.get("sources", {}).items():
        if source not in head["source_heads"] or not rows:
            raise ValueError("prepared posted member frame lost a source trajectory")
        if any(row.get("head_sha256") != head["source_heads"][source]
               or row.get("marker_sha256") != _published_digest(row.get("marker", {})) for row in rows):
            raise ValueError("prepared posted member frame changed a consumed source marker")


def validate_posted_member_head(binding, *, identity=None):
    """Pure portable validator; does not wait for or require future frames."""
    if not isinstance(binding, dict) or binding.get("schema") != HEAD_SCHEMA:
        raise ValueError("prepared posted member head schema differs")
    head = binding.get("head", {})
    if (head.get("schema") != HEAD_SCHEMA or binding.get("head_sha256") != _published_digest(head)
            or type(head.get("member_id")) is not int or type(head.get("seed")) is not int):
        raise ValueError("prepared posted member head lost its original index/seed/digest")
    times = tuple(_utc(datetime.fromisoformat(value)) for value in head.get("valid_times", ()))
    if not times or any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError("prepared posted member head lacks its increasing native time plan")
    _validate_frame(binding.get("initial"), head, 0, binding.get("initial_sha256"))
    if identity is not None:
        source = identity.get("source_identity", {})
        if "root_preparation" in source:
            source = source["root_preparation"]
        if not _source_matches(head["native"]["source"], source, head["native"]["input_plan_sha256"]):
            raise ValueError("prepared posted member source identity differs from its initialized native head")
    return binding


def validate_posted_member_segment(binding, head_binding, *, index):
    validate_posted_member_head(head_binding)
    head = head_binding["head"]
    if (not isinstance(binding, dict) or binding.get("schema") != SEGMENT_SCHEMA
            or binding.get("head_sha256") != head_binding["head_sha256"]
            or binding.get("index") != index or len(binding.get("frames", ())) != 2
            or len(binding.get("frame_sha256", ())) != 2):
        raise ValueError("posted member segment does not bind its exact two native endpoints")
    for position, frame, digest in zip((index, index + 1), binding["frames"], binding["frame_sha256"]):
        _validate_frame(frame, head, position, digest)
    return binding


__all__ = ["NativeSnapshotAuthority", "NativeSnapshotContext", "PostedPreparedMemberInput",
           "PostedMemberPreparation", "current_posted_preparation", "posted_preparation_scope",
           "replace_current_native_snapshot", "current_posted_domain", "bind_current_prepared_identity",
           "bind_current_source_identity", "posted_binding_from_identity",
           "validate_posted_member_head", "validate_posted_member_segment"]
