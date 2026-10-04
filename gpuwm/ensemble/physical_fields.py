"""Explicit interpretation of native physical arrays, separate from their names.

The second half of this module reads the physical field contracts of native
preparation implementations as data. Such a contract says what every array in
the implementation's physical store is: units, target-grid dimensions, vector
basis, the source fields it came from and the operation that produced it, plus
the vertical coordinate, the evidence a store must be given, and the sentences
a mislabelled store is refused with. Each contract is one packaged document,
pinned by SHA-256 in :mod:`gpuwm.source_authorities`. This module is the one
reader, so an implementation with the same capabilities needs a document and a
pin, not a module of its own. A mapped source declares no document: its
contract is derived from its own packaged mapping by
:mod:`gpuwm.ensemble.mapped_physical_contract`.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json

FIELD_SCHEMA = "gpuwm-ensemble-physical-fields.v1"


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_grid_sha256(grid):
    return hashlib.sha256(canonical_json(grid).encode()).hexdigest()


def field_contract_sha256(contract):
    return hashlib.sha256(canonical_json(contract).encode()).hexdigest()


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and not set(value) - set("0123456789abcdef")


def validate_field_contract(contract, grid):
    """Validate a caller's source-qualified units and coordinate declaration.

    No array spelling supplies a missing unit, vector basis or coordinate.
    Source adapters must supply these from their verified mapping contract.
    """
    if (not isinstance(contract, dict)
            or set(contract) != {"schema", "grid_sha256", "vertical", "arrays", "evidence"}
            or contract["schema"] != FIELD_SCHEMA
            or contract["grid_sha256"] != canonical_grid_sha256(grid)):
        raise ValueError("physical field contract lacks its schema or actual grid authority")
    vertical = contract["vertical"]
    if (not isinstance(vertical, dict)
            or set(vertical) != {"kind", "units", "values", "pressure_field"}
            or vertical["kind"] not in {"pressure_levels", "hybrid_model_levels", "representative_pressure_levels"}
            or vertical["values"] != "levels_hpa"
            or vertical["units"] != {"pressure_levels": "hPa", "hybrid_model_levels": "1",
                                      "representative_pressure_levels": "hPa"}[vertical["kind"]]
            or (vertical["pressure_field"] is not None and not isinstance(vertical["pressure_field"], str))
            or (vertical["kind"] != "pressure_levels" and vertical["pressure_field"] is None)):
        raise ValueError("physical vertical coordinate needs explicit pressure or hybrid authority")
    arrays, evidence = contract["arrays"], contract["evidence"]
    if not isinstance(evidence, dict) or not evidence or any(
            not isinstance(key, str) or not key or not _digest(value) for key, value in evidence.items()):
        raise ValueError("physical field contract requires hashed source mapping evidence")
    if not isinstance(arrays, dict) or not arrays or "levels_hpa" not in arrays:
        raise ValueError("physical field contract has no coordinate array declaration")
    for name, spec in arrays.items():
        if (not isinstance(name, str) or not name
                or not isinstance(spec, dict)
                or not {"units", "dimensions", "basis", "source_fields", "operation"} <= set(spec)
                or not set(spec) <= {"units", "dimensions", "basis", "source_fields", "operation", "source_units"}
                or not isinstance(spec["units"], str) or not spec["units"].strip()
                or not isinstance(spec["dimensions"], list) or not spec["dimensions"]
                or len(set(spec["dimensions"])) != len(spec["dimensions"])
                or not set(spec["dimensions"]) <= {"level", "soil_level", "y", "x", "y_stag", "x_stag"}
                or spec["basis"] not in {"scalar", "grid_x", "grid_y"}
                or not isinstance(spec["source_fields"], list) or not spec["source_fields"]
                or any(not isinstance(field, str) or not field for field in spec["source_fields"])
                or not isinstance(spec["operation"], str) or not spec["operation"].strip()):
            raise ValueError(f"physical field {name} lacks explicit units, dimensions or source operation")
        if "source_units" in spec and (not isinstance(spec["source_units"], str) or not spec["source_units"].strip()):
            raise ValueError(f"physical field {name} has an invalid source unit declaration")
        if spec["basis"] == "grid_x" and spec["dimensions"][-2:] != ["y", "x_stag"]:
            raise ValueError(f"physical grid-x vector {name} must use target x faces")
        if spec["basis"] == "grid_y" and spec["dimensions"][-2:] != ["y_stag", "x"]:
            raise ValueError(f"physical grid-y vector {name} must use target y faces")
    if arrays["levels_hpa"]["units"] != vertical["units"] or arrays["levels_hpa"]["dimensions"] != ["level"]:
        raise ValueError("physical level array differs from its vertical coordinate authority")
    pressure = vertical["pressure_field"]
    if pressure is not None and (pressure not in arrays or arrays[pressure]["units"] != "Pa"
                                 or arrays[pressure]["dimensions"] != ["level", "y", "x"]):
        raise ValueError("physical pressure coordinate must be an explicit mass-grid Pa field")
    shape = grid.get("mass_shape")
    if not isinstance(shape, list) or len(shape) != 2 or any(type(n) is not int or n < 1 for n in shape):
        raise ValueError("physical field contract requires the native mass-grid dimensions")
    return contract


def validate_field_inventory(contract, grid, inventory):
    validate_field_contract(contract, grid)
    if "levels_hpa" not in inventory or len(inventory["levels_hpa"]["shape"]) != 1:
        raise ValueError("physical frame has no one-dimensional native vertical coordinate")
    ny, nx = grid["mass_shape"]
    sizes = {"y": ny, "x": nx, "y_stag": ny+1, "x_stag": nx+1,
             "level": inventory["levels_hpa"]["shape"][0]}
    pressure = contract["vertical"]["pressure_field"]
    if pressure is not None and pressure not in inventory:
        raise ValueError("physical frame lacks its declared pressure coordinate")
    for name, item in inventory.items():
        if name not in contract["arrays"]:
            raise ValueError(f"physical array {name} has no source-qualified field contract")
        dims = contract["arrays"][name]["dimensions"]
        if len(dims) != len(item["shape"]) or any(
                type(n) is not int or n < 1 or (dim in sizes and n != sizes[dim])
                for dim, n in zip(dims, item["shape"])):
            raise ValueError(f"physical array {name} differs from its declared coordinate dimensions")
    return inventory


def native_field_attributes(spec):
    """Native attributes are checked independently of the file digest."""
    return {"units": spec["units"], "physical_dimensions": canonical_json(spec["dimensions"]),
            "physical_vector_basis": spec["basis"],
            "physical_source_fields": canonical_json(spec["source_fields"]),
            "physical_source_operation": spec["operation"],
            **({"physical_source_units": spec["source_units"]} if "source_units" in spec else {})}


# ---------------------------------------------------------------------------
# Packaged contracts of native preparation implementations
# ---------------------------------------------------------------------------

NATIVE_CONTRACT_SCHEMA = "gpuwm-native-physical-field-contract.v1"

#: The attributes a consumer compares. An operation sentence may be reworded
#: without changing what the numbers mean; these three may not.
COMPARED_ATTRIBUTES = ("units", "dimensions", "basis")

_CONTRACT_KEYS = {"schema", "contract_id", "description", "unit_authorities", "vertical",
                  "arrays", "evidence", "snapshot", "refusals"}
_REFUSALS = {"vertical", "unknown_array", "incompatible_array"}
_EVIDENCE_REFUSALS = {"evidence_type", "evidence_missing"}
_SNAPSHOT_REFUSALS = {"snapshot_levels", "snapshot_leading_axis"}

#: Parsed documents, keyed by contract and the verified bytes' digest.
_NATIVE_CONTRACTS: dict[str, tuple[str, Mapping]] = {}


def _validated_native_contract(contract_id, document):
    if (not isinstance(document, dict) or not _CONTRACT_KEYS <= set(document)
            or document["schema"] != NATIVE_CONTRACT_SCHEMA
            or document["contract_id"] != contract_id
            or not isinstance(document["arrays"], dict) or not document["arrays"]
            or not isinstance(document["vertical"], dict)
            or not isinstance(document["refusals"], dict)):
        raise RuntimeError(f"packaged physical field contract {contract_id} has an unsupported shape")
    needed = set(_REFUSALS)
    evidence, snapshot = document["evidence"], document["snapshot"]
    tables = set()
    if evidence is not None:
        if (not isinstance(evidence, dict)
                or set(evidence) != {"required", "any_of", "table_digests"}
                or not isinstance(evidence["table_digests"], dict)):
            raise RuntimeError(f"packaged physical field contract {contract_id} has unusable evidence rules")
        tables = set(evidence["table_digests"].values())
        needed |= _EVIDENCE_REFUSALS
    # The only keys beyond the fixed ones are the tables the evidence rules
    # bind by digest, and every table they name is present.
    if set(document) - _CONTRACT_KEYS != tables or tables & _CONTRACT_KEYS:
        raise RuntimeError(f"packaged physical field contract {contract_id} carries an unbound table")
    if snapshot is not None:
        if not isinstance(snapshot, dict) or set(snapshot) != {"levels", "leading_axis_length"}:
            raise RuntimeError(f"packaged physical field contract {contract_id} has unusable snapshot rules")
        needed |= _SNAPSHOT_REFUSALS
    if set(document["refusals"]) != needed or any(
            not isinstance(text, str) or not text.strip() for text in document["refusals"].values()):
        raise RuntimeError(f"packaged physical field contract {contract_id} does not state its refusals")
    return document


def native_contract_document(contract_id):
    """The verified document of one packaged native contract."""
    from gpuwm.source_authorities import packaged_physical_contract

    data = packaged_physical_contract(contract_id).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    cached = _NATIVE_CONTRACTS.get(contract_id)
    if cached is None or cached[0] != digest:
        _NATIVE_CONTRACTS[contract_id] = (
            digest, _validated_native_contract(contract_id, json.loads(data)))
    return _NATIVE_CONTRACTS[contract_id][1]


def native_contract_sha256(contract_id):
    """The pinned digest of the document that defines one native contract."""
    from gpuwm.source_authorities import packaged_physical_contract_sha256

    return packaged_physical_contract_sha256(contract_id)


def _native_refusal(document, key, **values):
    return ValueError(document["refusals"][key].format(**values))


def native_field_contract(contract_id, grid_identity, *, evidence):
    """Build one native implementation's field contract for a target grid.

    ``evidence`` binds the caller's own hashed authorities (decoder, mapper,
    source manifest or input plan). A contract that declares evidence rules
    refuses a caller that omits one. Tables the document carries are bound by
    digest under the evidence key the document names.
    """
    document = native_contract_document(contract_id)
    rules = document["evidence"]
    if rules is not None:
        if not isinstance(evidence, Mapping):
            raise _native_refusal(document, "evidence_type")
        present = set(evidence)
        if (not set(rules["required"]) <= present
                or any(not set(group) & present for group in rules["any_of"])):
            raise _native_refusal(document, "evidence_missing")
    evidence = dict(evidence)
    if rules is not None:
        for key, table in rules["table_digests"].items():
            evidence[key] = hashlib.sha256(json.dumps(
                document[table], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    contract = {"schema": FIELD_SCHEMA, "grid_sha256": canonical_grid_sha256(grid_identity),
                "vertical": dict(document["vertical"]),
                "arrays": deepcopy(document["arrays"]), "evidence": evidence}
    validate_field_contract(contract, grid_identity)
    return contract


def require_native_field_contract(contract_id, contract, grid_identity):
    """Refuse units, coordinates or bases the native consumer cannot read.

    Breakage it prevents: a store whose numbers are unchanged but whose
    declared units, vertical kind or wind basis differ would be initialized
    as if it were the native field, and the run would start from wrong
    physics with no error anywhere.
    """
    validate_field_contract(contract, grid_identity)
    document = native_contract_document(contract_id)
    if contract["vertical"] != document["vertical"]:
        raise _native_refusal(document, "vertical")
    expected = document["arrays"]
    for name, spec in contract["arrays"].items():
        reference = expected.get(name)
        if reference is None:
            raise _native_refusal(document, "unknown_array", name=name)
        for attribute in COMPARED_ATTRIBUTES:
            if spec[attribute] != reference[attribute]:
                raise _native_refusal(document, "incompatible_array", name=name, attribute=attribute)
    return contract


def validate_native_snapshot(contract_id, snapshot):
    """Hold a snapshot's coordinate values to the contract's native axes."""
    document = native_contract_document(contract_id)
    rules = document["snapshot"]
    if rules is None:
        return snapshot
    import numpy as np

    if not np.array_equal(snapshot.levels_hpa, np.asarray(rules["levels"])):
        raise _native_refusal(document, "snapshot_levels")
    for name, length in rules["leading_axis_length"].items():
        if name in snapshot.fields and snapshot.fields[name].shape[0] != length:
            raise _native_refusal(document, "snapshot_leading_axis", name=name)
    return snapshot
