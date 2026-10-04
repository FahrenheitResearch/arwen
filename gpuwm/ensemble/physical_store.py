"""Native, hash-bound physical snapshots before real initialization.

Rust writes and reads all numeric fields. Python owns only buffer staging,
the source/geometry receipts and immutable publication of the frame catalog.
"""
from __future__ import annotations

from datetime import datetime
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np

from gpuwm.ensemble.physical_fields import (
    FIELD_SCHEMA, canonical_grid_sha256, field_contract_sha256,
    native_field_attributes, validate_field_contract, validate_field_inventory,
)

LEGACY_SCHEMA = "gpuwm-ensemble-physical-store.v1"
SCHEMA = "gpuwm-ensemble-physical-store.v2"
QUALIFICATION_SCHEMA = "gpuwm-ensemble-physical-qualification.v1"
BINDING_SCHEMA = "gpuwm-ensemble-physical-input-binding.v1"
STATIC_SCHEMA = "gpuwm-ensemble-static-identity.v1"
_LAND_ATTRIBUTES = ("MMINLU", "NUM_LAND_CAT", "ISWATER", "ISLAKE", "ISICE", "ISURBAN", "ISOILWATER")
# Capture and initialization may run different installed implementations.
# Both complete identities remain in the receipt; these implementation
# records are distinct from the scientific source/mapping authorities.
_IMPLEMENTATION_SOURCE_KEYS = frozenset({
    "preprocessing", "native_bridge_sha256", "source_sha256", "identity_source",
    "git_commit", "git_tree", "git_status_short", "git_status_unknown_reason",
    "distribution_manifest_sha256", "installed_wheel", "installed_editable",
    "installed_source_content", "implementation_sha256", "git_source_identity",
})
_ARRAY_METADATA = ("water_temperature", "water_temperature_source", "soil_no_source_land")
_SCALAR_METADATA = (
    "specific_humidity_authority", "specific_humidity_undershoot_floor",
    "analyzed_species", "water_temperature_receipt", "horizontal_operators",
    "masked_field_repairs")


def _plain(value):
    if hasattr(value, "items"):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical(value):
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def _host(value):
    # Transfer and layout staging only. Numerical operations remain native.
    return np.ascontiguousarray(value.get() if hasattr(value, "get") else value)


def _land_attribute(name, value):
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(f"physical static attribute {name} must be scalar")
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    if name == "MMINLU":
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if not isinstance(value, str) or not value.strip(" \t\r\n\0"):
            raise ValueError("physical static MMINLU must name its land-class table")
        return value.strip(" \t\r\n\0")
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int,float))
            or not np.isfinite(value) or int(value) != value):
        raise ValueError(f"physical static attribute {name} must be an integer category")
    return int(value)


def physical_static_identity(static, attrs=None):
    """Hash the actual mapping statics, independently of container bytes.

    Numeric arrays use the native static cache's float64 representation,
    with fixed little-endian C storage. Shape is part of each field record.
    Land-category metadata is normalized separately from the numeric fields.
    """
    if not isinstance(static, Mapping) or not static:
        raise ValueError("physical mapping requires a nonempty static field mapping")
    if attrs is not None and not isinstance(attrs, Mapping):
        raise ValueError("physical static attributes must be a mapping")
    attributes, fields = {}, {}
    for mapping in (static, {} if attrs is None else attrs):
        for name in _LAND_ATTRIBUTES:
            if name in mapping and mapping[name] is not None:
                value = _land_attribute(name, mapping[name])
                if name in attributes and attributes[name] != value:
                    raise ValueError(f"physical static attribute {name} has conflicting authorities")
                attributes[name] = value
    for name,value in sorted(static.items()):
        if not isinstance(name,str) or not name:
            raise ValueError("physical static field names must be nonempty strings")
        if name in _LAND_ATTRIBUTES:
            continue
        array = _host(value)
        if array.dtype.kind not in "fiub":
            continue
        array = np.ascontiguousarray(array, dtype=np.dtype("<f8"))
        if not np.isfinite(array).all() or array.size == 0:
            raise ValueError(f"physical static field {name} must be nonempty and finite")
        fields[name] = {"shape": list(array.shape), "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
    if not fields:
        raise ValueError("physical mapping has no numerical static fields")
    return {"schema": STATIC_SCHEMA, "fields": fields, "attributes": attributes}


def _valid_digest(value):
    return isinstance(value,str) and len(value) == 64 and not set(value) - set("0123456789abcdef")


def _validate_static_identity(identity):
    if (not isinstance(identity,dict) or set(identity) != {"schema","fields","attributes"}
            or identity["schema"] != STATIC_SCHEMA or not isinstance(identity["fields"],dict)
            or not identity["fields"] or not isinstance(identity["attributes"],dict)):
        raise ValueError("physical input lacks the captured static mapping identity; recapture its base")
    for name,item in identity["fields"].items():
        if (not isinstance(name,str) or not name or not isinstance(item,dict)
                or set(item) != {"shape","sha256"} or not isinstance(item["shape"],list)
                or any(type(length) is not int or length < 1 for length in item["shape"])
                or not _valid_digest(item["sha256"])):
            raise ValueError("physical input static field identity is malformed")
    for name,value in identity["attributes"].items():
        if name not in _LAND_ATTRIBUTES or _land_attribute(name,value) != value:
            raise ValueError("physical input land-category identity is malformed")
    return identity


def _semantic_source_identity(source):
    if not isinstance(source,dict) or not source:
        raise ValueError("physical input needs its ordinary prepared source authority")
    excluded = _IMPLEMENTATION_SOURCE_KEYS | {"input_manifest_sha256", "static_identity", "ensemble_physical_input",
                                             "ensemble_posted_physical_input"}
    return {key:value for key,value in source.items() if key not in excluded}


class NativePhysicalStore:
    """One source trajectory on one common horizontal grid.

    Writers supply actual native geometry and source receipts, including the
    verified input manifest. A sealed reader checks each frame's native file
    before returning its arrays. Stores never invent a missing valid time.
    """
    def __init__(self, root, *, grid_identity=None, source_identity=None,
                 field_contract=None, allow_unqualified_legacy=False):
        self.root = Path(root).resolve()
        self.manifest_path = self.root / "physical-store.json"
        self.qualification = None
        self._allow_unqualified_legacy = allow_unqualified_legacy
        if self.manifest_path.exists():
            if grid_identity is not None or source_identity is not None or field_contract is not None:
                raise FileExistsError("a sealed physical store cannot be reopened for writing")
            manifest_bytes = self.manifest_path.read_bytes()
            self.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
            self.document = json.loads(manifest_bytes)
            if self.document.get("schema") not in {SCHEMA, LEGACY_SCHEMA}:
                raise ValueError("physical store schema differs from the native snapshot contract")
            self.writable = False
            if self.document["schema"] == SCHEMA:
                validate_field_contract(self.document.get("field_contract"), self.document["grid"])
            else:
                certificate = self.root / "physical-qualification.json"
                if certificate.exists():
                    self.qualification = json.loads(certificate.read_text(encoding="utf-8"))
                    _validate_qualification(self.document, self.qualification,
                                            self.manifest_sha256)
                elif not allow_unqualified_legacy:
                    raise ValueError("legacy physical store has no source-qualified units and coordinates; qualify or recapture it")
        else:
            if not isinstance(grid_identity, dict) or not grid_identity:
                raise ValueError("a physical store needs the actual native target geometry")
            if not isinstance(source_identity, dict) or not source_identity:
                raise ValueError("a physical store needs its verified native source receipt")
            validate_field_contract(field_contract, grid_identity)
            self.root.mkdir(parents=True, exist_ok=True)
            self.document = {"schema": SCHEMA, "grid": _plain(grid_identity),
                             "source": _plain(source_identity), "frames": [],
                             "field_contract": _plain(field_contract)}
            self.writable = True

    @property
    def field_contract(self):
        if self.document["schema"] == SCHEMA:
            return self.document["field_contract"]
        return None if self.qualification is None else self.qualification["field_contract"]

    def require_field_contract(self):
        return validate_field_contract(self.field_contract, self.document["grid"])

    @property
    def grid_sha256(self):
        return hashlib.sha256(_canonical(self.document["grid"]).encode()).hexdigest()

    @property
    def times(self):
        return tuple(datetime.fromisoformat(frame["valid_time"]) for frame in self.document["frames"])

    def write(self, snapshot):
        from gpuwm.io.nc_writer_bridge import ClassicSchema
        if not self.writable:
            raise ValueError("a sealed physical store is immutable")
        valid_time = snapshot.valid_time.isoformat()
        if self.document["frames"] and snapshot.valid_time <= self.times[-1]:
            raise ValueError("physical snapshots must have strictly increasing valid times")
        name = f"frame-{len(self.document['frames']):04d}.nc"
        final = self.root / name
        if final.exists():
            raise FileExistsError(final)
        temporary = self.root / f".{name}.{uuid.uuid4().hex}.partial"
        arrays = {"levels_hpa": _host(snapshot.levels_hpa)}
        arrays.update({"field__"+key: _host(value) for key, value in snapshot.fields.items()})
        for key in _ARRAY_METADATA:
            value = getattr(snapshot, key)
            if value is not None:
                arrays["meta__"+key] = _host(value)
        schema = ClassicSchema("cdf5")
        schema.put_global_attr("physical_store_schema", SCHEMA)
        schema.put_global_attr("valid_time", valid_time)
        schema.put_global_attr("grid_sha256", self.grid_sha256)
        schema.put_global_attr("field_contract_sha256", field_contract_sha256(self.require_field_contract()))
        schema.put_global_attr("physical_vertical_coordinate", _canonical(self.field_contract["vertical"]))
        variables, inventory, dimensions = {}, {}, {}
        for key, array in sorted(arrays.items()):
            if array.dtype.kind not in "fiub" or not np.isfinite(array).all():
                raise ValueError(f"physical field {key} must be finite numeric native storage")
            dtype = np.dtype("u1") if array.dtype.kind == "b" else array.dtype
            dims = []
            for axis, length in enumerate(array.shape):
                dim_key = (axis, int(length))
                if dim_key not in dimensions:
                    dimensions[dim_key] = schema.def_dim(f"axis{axis}_{length}", int(length))
                dims.append(dimensions[dim_key])
            variables[key] = schema.def_var(key, dtype, dims)
            inventory[key] = {"shape": list(array.shape), "dtype": array.dtype.str}
        validate_field_inventory(self.field_contract, self.document["grid"], inventory)
        for key, varid in variables.items():
            for attribute, value in native_field_attributes(self.field_contract["arrays"][key]).items():
                schema.put_var_attr(varid, attribute, value)
        with schema.create(temporary) as writer:
            for key, varid in variables.items():
                value = arrays[key]
                writer.write_var(varid, value.astype("u1") if value.dtype.kind == "b" else value)
        temporary.replace(final)
        frame = {"file": name, "sha256": digest_file(final), "bytes": final.stat().st_size,
                 "valid_time": valid_time, "arrays": inventory,
                 "metadata": {key: _plain(getattr(snapshot, key)) for key in _SCALAR_METADATA}}
        self.document["frames"].append(frame)
        return frame

    def seal(self):
        if not self.writable or not self.document["frames"]:
            raise ValueError("only a nonempty writable physical store can be sealed")
        # Exclusive publication avoids replacing any previous trajectory.
        with self.manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(_canonical(self.document)+"\n")
        self.writable = False
        self.manifest_sha256 = digest_file(self.manifest_path)
        return {"path": str(self.manifest_path), "sha256": digest_file(self.manifest_path),
                "grid_sha256": self.grid_sha256, "frames": len(self.times)}

    def read(self, index):
        from gpuwm.ingest.horiz import HorizontalSnapshot
        from gpuwm.netcdf_bridge import open_dataset
        if self.writable:
            raise ValueError("a physical store must be sealed before it supplies a forecast")
        if digest_file(self.manifest_path) != self.manifest_sha256:
            raise ValueError("physical store manifest changed after its authority was loaded")
        frame = self.document["frames"][index]
        contract = self.field_contract
        if contract is not None:
            validate_field_inventory(contract, self.document["grid"], frame["arrays"])
        elif not self._allow_unqualified_legacy:
            self.require_field_contract()
        path = (self.root / frame["file"]).resolve()
        if path.parent != self.root:
            raise ValueError("physical frame escapes its sealed store")
        if path.stat().st_size != frame["bytes"] or digest_file(path) != frame["sha256"]:
            raise ValueError("physical frame bytes differ from the sealed native receipt")
        arrays = {}
        with open_dataset(path) as dataset:
            expected_headers = {"physical_store_schema": self.document["schema"], "valid_time": frame["valid_time"],
                                "grid_sha256": self.grid_sha256}
            if self.document["schema"] == SCHEMA:
                expected_headers.update(field_contract_sha256=field_contract_sha256(contract),
                                        physical_vertical_coordinate=_canonical(contract["vertical"]))
            if any(dataset.global_attributes.get(key) != value for key,value in expected_headers.items()):
                raise ValueError("native physical frame header differs from its valid time or grid receipt")
            if set(dataset.variables) != set(frame["arrays"]):
                raise ValueError("native physical frame inventory differs from its receipt")
            dataset.set_auto_maskandscale(False)
            for key, item in frame["arrays"].items():
                variable = dataset.variables[key]
                if self.document["schema"] == SCHEMA and any(
                        variable.attributes.get(name) != value for name, value in
                        native_field_attributes(contract["arrays"][key]).items()):
                    raise ValueError(f"native physical field {key} units or coordinates differ from its receipt")
                recorded_dtype = np.dtype(item["dtype"])
                storage_dtype = np.dtype("u1") if recorded_dtype.kind == "b" else recorded_dtype
                if (variable.dtype.newbyteorder("=") != storage_dtype.newbyteorder("=")
                        or list(variable.shape) != item["shape"]):
                    raise ValueError("native physical frame dtype or dimensions differ from its receipt")
                value = np.ascontiguousarray(variable[:], dtype=recorded_dtype)
                if list(value.shape) != item["shape"]:
                    raise ValueError("native physical frame dimensions differ from its receipt")
                arrays[key] = value
        metadata = dict(frame["metadata"])
        if metadata["analyzed_species"] is not None:
            metadata["analyzed_species"] = tuple(metadata["analyzed_species"])
        return HorizontalSnapshot(
            valid_time=datetime.fromisoformat(frame["valid_time"]),
            levels_hpa=arrays["levels_hpa"],
            fields={key[len("field__"):]: value for key, value in arrays.items() if key.startswith("field__")},
            **{key: arrays.get("meta__"+key) for key in _ARRAY_METADATA}, **metadata)


def validate_physical_input_binding(binding, *, input_manifest_sha256,
                                    source_identity=None, static_identity=None):
    """Validate the portable authority embedded in a prepared cache identity."""
    if not isinstance(binding, dict) or set(binding) not in (
            {"schema", "manifest_sha256", "manifest"},
            {"schema", "manifest_sha256", "manifest", "qualification"}):
        raise ValueError("prepared physical input binding is incomplete")
    if binding["schema"] != BINDING_SCHEMA:
        raise ValueError("prepared physical input binding schema is unsupported")
    document = binding["manifest"]
    if (not isinstance(document, dict) or document.get("schema") not in {SCHEMA, LEGACY_SCHEMA}
            or set(document) != ({"schema", "source", "grid", "frames", "field_contract"}
                                 if document.get("schema") == SCHEMA else {"schema", "source", "grid", "frames"})
            or not isinstance(document["source"], dict)
            or not isinstance(document["grid"], dict) or not document["grid"]
            or not isinstance(document["frames"], list) or not document["frames"]):
        raise ValueError("prepared physical input manifest is incomplete")
    observed = hashlib.sha256((_canonical(document)+"\n").encode()).hexdigest()
    if document["schema"] == LEGACY_SCHEMA and observed != binding["manifest_sha256"]:
        # The original text writer used the host newline. Qualified v1
        # stores keep those exact bytes, including Windows CRLF.
        observed = hashlib.sha256((_canonical(document)+"\r\n").encode()).hexdigest()
    if observed != binding["manifest_sha256"]:
        raise ValueError("prepared physical input manifest differs from its bound digest")
    if document["schema"] == SCHEMA:
        if "qualification" in binding:
            raise ValueError("native v2 physical store cannot use a legacy qualification")
        contract = validate_field_contract(document["field_contract"], document["grid"])
    else:
        certificate = binding.get("qualification")
        _validate_qualification(document, certificate, observed)
        contract = certificate["field_contract"]
    source = document["source"]
    if source.get("schema") == "gpuwm-ensemble-recentered-preparation.v1":
        if not isinstance(source.get("donors"), dict) or len(source["donors"]) < 2:
            raise ValueError("recentered input has no fixed donor population")
        if source.get("selected_member") not in source["donors"]:
            raise ValueError("recentered input selected member is outside its donor population")
        base = source.get("base", {}).get("source", {})
    else:
        base = source
    if base.get("input_manifest_sha256") != input_manifest_sha256:
        raise ValueError("physical input base source differs from the prepared source manifest")
    captured_static = _validate_static_identity(base.get("static_identity"))
    if static_identity is not None:
        if _canonical(captured_static) != _canonical(_validate_static_identity(static_identity)):
            raise ValueError("physical input statics differ from the actual native mapping statics")
    if source_identity is not None:
        if _canonical(_semantic_source_identity(base)) != _canonical(_semantic_source_identity(source_identity)):
            raise ValueError("physical input base source authority differs from the ordinary prepared source identity")
    times = []
    for frame in document["frames"]:
        if not isinstance(frame, dict) or not {"file", "sha256", "bytes", "valid_time", "arrays", "metadata"} <= set(frame):
            raise ValueError("physical input frame identity is incomplete")
        digest = frame["sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("physical input frame digest is malformed")
        times.append(datetime.fromisoformat(frame["valid_time"]))
        validate_field_inventory(contract, document["grid"], frame["arrays"])
    if times != sorted(set(times)):
        raise ValueError("physical input frame valid times must increase without duplicates")
    return binding


def physical_input_binding(store, grid, cfg, source_identity, *, input_manifest_sha256,
                           static_identity=None):
    """Bind a checked replacement to the actual geometry and base authority."""
    from gpuwm.native_wrf_contract import native_geometry_contract
    if not isinstance(store, NativePhysicalStore) or store.writable:
        raise ValueError("physical input must be a sealed native store")
    if store.document["grid"] != native_geometry_contract(grid, cfg):
        raise ValueError("physical input geometry differs from the prepared model grid")
    if not isinstance(source_identity, dict) or not source_identity:
        raise ValueError("physical input needs the ordinary prepared source authority")
    if static_identity is None:
        raise ValueError("physical input must verify the actual native mapping static identity")
    if digest_file(store.manifest_path) != store.manifest_sha256:
        raise ValueError("physical store manifest changed after its authority was loaded")
    binding = {"schema": BINDING_SCHEMA, "manifest_sha256": store.manifest_sha256,
               "manifest": store.document}
    store.require_field_contract()
    if store.qualification is not None:
        binding["qualification"] = store.qualification
    return validate_physical_input_binding(binding, input_manifest_sha256=input_manifest_sha256,
                                            source_identity=source_identity, static_identity=static_identity)


def _validate_qualification(document, certificate, manifest_sha256):
    expected_keys = {"schema", "original_manifest_sha256", "field_contract", "source_identity_sha256",
                     "frames", "evidence", "native_attributes_present"}
    if (not isinstance(certificate, dict) or set(certificate) != expected_keys
            or certificate["schema"] != QUALIFICATION_SCHEMA
            or certificate["original_manifest_sha256"] != manifest_sha256
            or certificate["native_attributes_present"] is not False
            or certificate["source_identity_sha256"] != hashlib.sha256(_canonical(document["source"]).encode()).hexdigest()):
        raise ValueError("legacy physical qualification differs from its exact source manifest")
    contract = validate_field_contract(certificate["field_contract"], document["grid"])
    frames = [{key: frame[key] for key in ("file", "sha256", "bytes")} for frame in document["frames"]]
    if not frames or certificate["frames"] != frames:
        raise ValueError("legacy physical qualification differs from its exact native frame bytes")
    evidence = certificate["evidence"]
    if not isinstance(evidence, dict) or set(evidence) != set(contract["evidence"]):
        raise ValueError("legacy physical qualification lacks source revalidation evidence")
    for role, expected in contract["evidence"].items():
        item = evidence[role]
        if (not isinstance(item, dict) or set(item) != {"path", "sha256", "bytes"}
                or not isinstance(item["path"], str) or not item["path"]
                or item["sha256"] != expected or type(item["bytes"]) is not int or item["bytes"] < 1):
            raise ValueError("legacy physical qualification source evidence differs from its field authority")
    for frame in document["frames"]:
        validate_field_inventory(contract, document["grid"], frame["arrays"])
    return certificate


def qualify_legacy_store(root, *, expected_manifest_sha256, field_contract,
                         source_identity, evidence_files):
    """Publish a source-qualified certificate without modifying legacy arrays.

    The source adapter must first revalidate its native mapping contract and
    supply the exact evidence files that define that contract. This operation
    rechecks those files, the captured source identity, every native frame,
    and all declared dimensions. Original manifests and arrays stay intact.
    """
    store = NativePhysicalStore(root, allow_unqualified_legacy=True)
    if store.document["schema"] != LEGACY_SCHEMA:
        raise ValueError("only legacy physical stores need a qualification certificate")
    if store.qualification is not None:
        raise FileExistsError("legacy physical store already has an immutable qualification")
    if not _valid_digest(expected_manifest_sha256) or digest_file(store.manifest_path) != expected_manifest_sha256:
        raise ValueError("legacy physical store differs from the expected original manifest")
    if _canonical(source_identity) != _canonical(store.document["source"]):
        raise ValueError("legacy physical qualification needs the exact verified capture source identity")
    validate_field_contract(field_contract, store.document["grid"])
    if set(evidence_files) != set(field_contract["evidence"]):
        raise ValueError("legacy qualification requires every declared source evidence file")
    evidence = {}
    for role, expected in field_contract["evidence"].items():
        path = Path(evidence_files[role]).resolve()
        if not path.is_file() or digest_file(path) != expected:
            raise ValueError(f"legacy qualification source evidence changed: {role}")
        evidence[role] = {"path": str(path), "sha256": expected, "bytes": path.stat().st_size}
    for index, frame in enumerate(store.document["frames"]):
        validate_field_inventory(field_contract, store.document["grid"], frame["arrays"])
        store.read(index)
    if digest_file(store.manifest_path) != expected_manifest_sha256 or any(
            digest_file(item["path"]) != item["sha256"] for item in evidence.values()):
        raise ValueError("legacy physical source authority changed during qualification")
    certificate = {"schema": QUALIFICATION_SCHEMA,
                   "original_manifest_sha256": expected_manifest_sha256,
                   "source_identity_sha256": hashlib.sha256(_canonical(store.document["source"]).encode()).hexdigest(),
                   "field_contract": _plain(field_contract), "native_attributes_present": False,
                   "frames": [{key: frame[key] for key in ("file", "sha256", "bytes")}
                              for frame in store.document["frames"]], "evidence": evidence}
    _validate_qualification(store.document, certificate, expected_manifest_sha256)
    path = store.root / "physical-qualification.json"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(_canonical(certificate)+"\n")
    return {"path": str(path), "sha256": digest_file(path),
            "original_manifest_sha256": expected_manifest_sha256,
            "field_contract_sha256": field_contract_sha256(field_contract), "frames": len(store.times)}


__all__ = ["NativePhysicalStore", "SCHEMA", "LEGACY_SCHEMA", "FIELD_SCHEMA", "digest_file",
           "physical_input_binding", "validate_physical_input_binding", "physical_static_identity",
           "canonical_grid_sha256", "validate_field_contract", "qualify_legacy_store"]
