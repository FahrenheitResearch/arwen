"""A separately declared analysis for a mapped source's lateral forcing.

Source names select packaged tables. Decode, horizontal mapping and WRF-real
initialization remain the same operators used for every forcing frame.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile


REQUEST_SCHEMA = "gpuwm-initial-source-v1"
RECEIPT_SCHEMA = "gpuwm-initial-source-receipt-v1"


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _request_document(doc):
    """Validate declaration shape without requiring the original data files."""
    if (not isinstance(doc, dict)
            or set(doc) != {"schema", "source", "input_files", "supplements"}
            or doc["schema"] != REQUEST_SCHEMA):
        raise ValueError("initial inputs must declare schema, source, input_files and supplements")
    if not isinstance(doc["source"], str):
        raise ValueError("initial source must name a packaged source")
    if (not isinstance(doc["supplements"], dict)
            or any(not isinstance(k, str) or not k for k in doc["supplements"])):
        raise ValueError("initial supplements must map role names to path lists")
    for values in (doc["input_files"], *doc["supplements"].values()):
        if (not isinstance(values, list) or not values
                or any(not isinstance(v, str) or not v for v in values)):
            raise ValueError("initial input inventories must be nonempty path lists")
    return doc


def _is_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _evidence_name(name):
    # Check both path separators regardless of the host. A portable bundle
    # must not turn a harmless Unix filename into a path or drive on Windows.
    if (not isinstance(name, str) or not name or name in {".", ".."}
            or Path(name).name != name or any(c in name for c in ("\\", ":", "\x00"))):
        raise ValueError("initial analysis evidence must stay inside its evidence directory")
    return name


def _json_object(payload, label):
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"initial analysis {label} must be an object")
    return value


def _decoder_digests(value):
    if (not isinstance(value, dict) or not value
            or any(not isinstance(role, str) or not role
                   or not isinstance(row, dict) or not _is_sha256(row.get("sha256"))
                   for role, row in value.items())):
        raise ValueError("initial analysis decoder inventory needs a SHA-256 for every binary")
    return {role: row["sha256"] for role, row in value.items()}


def read_initial_inputs(path):
    """Resolve one analysis inventory relative to its declaration."""
    path = Path(path).resolve()
    payload = path.read_bytes()
    doc = _request_document(json.loads(payload))
    from gpuwm.source_adapters import get_source_adapter
    adapter = get_source_adapter(doc["source"])
    if adapter.packaged_profile is None:
        raise ValueError("initial source needs a packaged mapping for the shared mapped decoder")

    def paths(values):
        result = tuple((path.parent / v).resolve() for v in values)
        if len(set(result)) != len(result):
            raise ValueError("initial input inventory repeats a source file")
        for value in result:
            if not value.is_file():
                raise ValueError(f"initial source file is missing: {value}")
        return result

    return {
        "path": path, "payload": payload, "source": adapter.source_id,
        "profile": adapter.packaged_profile,
        "primary": paths(doc["input_files"]),
        "supplements": {role: paths(values) for role, values in doc["supplements"].items()},
    }


@dataclass
class InitialAnalysis:
    snapshot: object
    mapping: dict
    soil_layer_contract: object
    receipt: dict
    evidence: dict


@contextmanager
def decode_initial_analysis(request, *, output_parent, grids, valid_time,
                            workers=None, decoders=None):
    """Decode exactly one valid time and retain its portable authorities."""
    from gpuwm.mapped_authoring import author_input_manifest
    from gpuwm.mapped_composition import decode_composed_source, mapped_composition_receipt
    from gpuwm.mapped_source import load_mapping
    from gpuwm.source_authorities import (
        packaged_authorities, packaged_contributing_mappings, packaged_profile,
    )
    profile = packaged_profile(request["profile"])
    authorities = packaged_authorities(request["profile"])
    contributing = packaged_contributing_mappings(request["profile"])
    provenance = {str(profile["provenance_role"]): authorities["provenance"]}
    decoder_args = {k: (decoders or {}).get(k) for k in (
        "grib1_bridge", "grib2_inventory", "grib2_dump")}
    Path(output_parent).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".initial-", dir=output_parent) as folder:
        manifest = Path(folder) / "input-manifest.json"
        common = dict(primary_files=request["primary"],
                      supplement_files=request["supplements"],
                      provenance_files=provenance, contributing_mappings=contributing,
                      **decoder_args)
        author_input_manifest(manifest, mapping_path=authorities["mapping"],
                              composition_path=authorities["composition"], **common)
        manifest_bytes = manifest.read_bytes()
        bundle = decode_composed_source(
            authorities["composition"], authorities["mapping"],
            request["primary"], request["supplements"], provenance,
            input_manifest=manifest, input_manifest_sha256=_sha(manifest_bytes),
            contributing_mappings=contributing, scratch_destination=folder,
            atmospheric_grids=grids, workers=workers, **decoder_args)
        try:
            snapshots = bundle.regular_snapshots()
            if hasattr(snapshots, "for_grids"):
                snapshots = snapshots.for_grids(grids)
            times = getattr(snapshots, "valid_times", None)
            if times is None:
                times = tuple(s.valid_time for s in snapshots)
            if tuple(times) != (valid_time,):
                raise ValueError(
                    f"initial analysis must contain exactly {valid_time.isoformat()}, got {tuple(times)}")
            if request["path"].read_bytes() != request["payload"]:
                raise ValueError("initial input declaration changed during decode")
            composition_receipt = mapped_composition_receipt(bundle)
            evidence = {
                "request.json": request["payload"],
                "mapping.json": Path(authorities["mapping"]).read_bytes(),
                "composition.json": Path(authorities["composition"]).read_bytes(),
                "provenance.json": Path(authorities["provenance"]).read_bytes(),
                "input-manifest.json": manifest_bytes,
                "composition-receipt.json": (json.dumps(composition_receipt,
                    indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
            }
            for index, (_, value) in enumerate(sorted(contributing.items())):
                evidence[f"contributing-{index}.json"] = Path(value).read_bytes()
            receipt = {
                "schema": RECEIPT_SCHEMA, "source": request["source"],
                "valid_time": valid_time.isoformat(),
                "evidence": {name: _sha(data) for name, data in sorted(evidence.items())},
            }
            yield InitialAnalysis(snapshots[0], load_mapping(authorities["mapping"]),
                                  bundle.soil_layer_contract, receipt, evidence)
        finally:
            bundle.close()


def write_initial_evidence(evidence_root, evidence):
    for name in evidence:
        _evidence_name(name)
    target = Path(evidence_root) / "initial"
    target.mkdir()
    for name, payload in evidence.items():
        (target / name).write_bytes(payload)


def validate_initial_evidence(prepared_root, receipt, *, valid_time):
    """Verify donor authorities without requiring the original raw data."""
    from gpuwm.source_adapters import get_source_adapter
    from gpuwm.source_authorities import (
        packaged_authority_sha256, packaged_contributing_sha256, packaged_profile,
    )
    required_receipt = {"schema", "source", "valid_time", "evidence", "aerosol_source"}
    if (not isinstance(receipt, dict)
            or not required_receipt <= set(receipt)
            or set(receipt) - required_receipt - {"boundary_head_preprocessing"}
            or receipt["schema"] != RECEIPT_SCHEMA
            or receipt["valid_time"] != valid_time
            or not isinstance(receipt["source"], str)
            or not isinstance(receipt["aerosol_source"], str)
            or receipt["aerosol_source"] not in {"initial-analysis", "boundary-analysis"}):
        raise ValueError("initial analysis receipt is incomplete or has another valid time")
    if "boundary_head_preprocessing" in receipt:
        phase = receipt["boundary_head_preprocessing"]
        if not isinstance(phase, dict) or phase.get("backend") not in ("cpu", "cuda"):
            raise ValueError("initial analysis boundary-head preprocessing must name a CPU or CUDA backend")
        try:
            json.dumps(phase, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ValueError("initial analysis boundary-head preprocessing contains invalid metadata") from error
    adapter = get_source_adapter(receipt["source"])
    if adapter.packaged_profile is None:
        raise ValueError("initial analysis receipt does not name a packaged mapping")
    pins = packaged_authority_sha256(adapter.packaged_profile)
    prepared = Path(prepared_root).resolve()
    root = (prepared / "source-evidence" / "initial").resolve()
    if not root.is_relative_to(prepared):
        raise ValueError("initial analysis evidence must stay inside its prepared directory")
    files = receipt["evidence"]
    required = {"request.json", "mapping.json", "composition.json", "provenance.json",
                "input-manifest.json", "composition-receipt.json"}
    contributing = {f"contributing-{index}.json": digest for index, (_, digest)
                    in enumerate(sorted(packaged_contributing_sha256(adapter.packaged_profile).items()))}
    required.update(contributing)
    if not isinstance(files, dict) or not required <= set(files):
        raise ValueError("initial analysis evidence inventory is incomplete")
    for name, digest in files.items():
        _evidence_name(name)
        if not _is_sha256(digest):
            raise ValueError(f"initial analysis evidence needs a SHA-256: {name}")
        path = (root / name).resolve()
        if path.parent != root or not path.is_file() or _sha(path.read_bytes()) != digest:
            raise ValueError(f"initial analysis evidence changed or is missing: {name}")
    for role in ("mapping", "composition", "provenance"):
        if files[f"{role}.json"] != pins[role]:
            raise ValueError(f"initial analysis {role} differs from its packaged authority")
    for name, expected in contributing.items():
        if files[name] != expected:
            raise ValueError(f"initial analysis {name} differs from its packaged authority")
    manifest = _json_object((root / "input-manifest.json").read_bytes(), "input manifest")
    if any(manifest.get(f"{role}_sha256") != files[f"{role}.json"]
           for role in ("mapping", "composition")):
        raise ValueError("initial analysis input manifest binds different authorities")
    provenance = manifest.get("provenance")
    provenance_role = str(packaged_profile(adapter.packaged_profile)["provenance_role"])
    provenance_row = provenance.get(provenance_role) if isinstance(provenance, dict) else None
    if not isinstance(provenance_row, dict) or provenance_row.get("sha256") != files["provenance.json"]:
        raise ValueError("initial analysis input manifest binds different provenance")
    from gpuwm.mapped_composition import (
        INPUT_MANIFEST_SCHEMA, RECEIPT_SCHEMA as COMPOSITION_RECEIPT_SCHEMA,
        _canonical_sha256,
    )
    request = _request_document(json.loads((root / "request.json").read_bytes()))
    if get_source_adapter(request["source"]).source_id != adapter.source_id:
        raise ValueError("initial analysis declaration names another source")
    primary = manifest.get("primary_files")
    if (manifest.get("schema") != INPUT_MANIFEST_SCHEMA
            or not isinstance(primary, list) or not primary
            or len(primary) != len(request["input_files"])
            or any(not isinstance(row, dict) or not _is_sha256(row.get("sha256")) for row in primary)):
        raise ValueError("initial analysis manifest has no bound source inventory")
    supplements = manifest.get("supplements")
    if (not isinstance(supplements, dict) or set(supplements) != set(request["supplements"])
            or any(not isinstance(rows, list) or len(rows) != len(request["supplements"][role])
                   or any(not isinstance(row, dict) or not _is_sha256(row.get("sha256")) for row in rows)
                   for role, rows in supplements.items())):
        raise ValueError("initial analysis manifest has different or unbound supplements")
    composed = _json_object((root / "composition-receipt.json").read_bytes(), "composition receipt")
    if composed.get("schema") != COMPOSITION_RECEIPT_SCHEMA:
        raise ValueError("initial analysis composition receipt has another schema")
    content = {key: value for key, value in composed.items() if key != "receipt_content_sha256"}
    if composed.get("receipt_content_sha256") != _canonical_sha256(content):
        raise ValueError("initial analysis composition receipt content differs from its digest")
    for role, name in (("mapping", "mapping.json"), ("composition", "composition.json"),
                       ("input_manifest", "input-manifest.json")):
        identity = composed.get(role)
        if not isinstance(identity, dict) or identity.get("sha256") != files[name]:
            raise ValueError(f"initial analysis composition receipt binds a different {role}")
    if (type(composed.get("frame_count")) is not int or composed["frame_count"] != 1
            or composed.get("valid_times") != [valid_time]
            or not isinstance(composed.get("frames"), list) or len(composed["frames"]) != 1):
        raise ValueError("initial analysis composition receipt has another valid time or frame count")
    frame = composed["frames"][0]
    if (not isinstance(frame, dict) or not _is_sha256(frame.get("header_sha256"))
            or not _is_sha256(frame.get("terrain_sha256"))
            or type(frame.get("field_count")) is not int or frame["field_count"] <= 0):
        raise ValueError("initial analysis composition receipt has no bound frame")
    manifest_decoders, receipt_decoders = manifest.get("decoders"), composed.get("decoders")
    if _decoder_digests(manifest_decoders) != _decoder_digests(receipt_decoders):
        raise ValueError("initial analysis composition and manifest name different decoders")
    return receipt
