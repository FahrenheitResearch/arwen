"""Physical source frames consumed before a source trajectory has sealed.

Each ready marker binds one sealed native frame, the immutable source plan,
and the posted/decoded evidence the ordinary source adapter verified. There
is no second fetch loop here. A missing frame either invokes the caller's
existing posted-lead wait/capture function or raises ``PhysicalFramePending``.
Recentring uses the same native operators as complete physical stores.
"""
from __future__ import annotations

from bisect import bisect_left
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import uuid

from gpuwm.ensemble.physical_fields import validate_field_contract
from gpuwm.ensemble.physical_store import NativePhysicalStore, digest_file
from gpuwm.ensemble.recipes import RECIPE_KINDS, RecipeMember, SourceRecipe, SourceTrajectory

HEAD_SCHEMA = "gpuwm-ensemble-posted-physical-head.v1"
FRAME_SCHEMA = "gpuwm-ensemble-posted-physical-frame.v1"
SEAL_SCHEMA = "gpuwm-ensemble-posted-physical-seal.v1"
PROVIDER_SCHEMA = "gpuwm-ensemble-posted-physical-provider.v1"
PROVIDER_PLAN_SCHEMA = "gpuwm-ensemble-posted-physical-plan.v1"
PLAN_REFERENCE_SCHEMA = "gpuwm-ensemble-posted-plan-reference.v1"
SOURCE_AUTHORITY_SCHEMA = "gpuwm-ensemble-posted-source-authority.v1"
INPUT_BINDING_SCHEMA = "gpuwm-ensemble-posted-physical-input.v1"
SOURCE_SEAL_SCHEMA = "gpuwm-ensemble-posted-source-seal.v1"
PROVIDER_SEAL_SCHEMA = "gpuwm-ensemble-posted-provider-seal.v1"
SOURCE_AUTHORITY_KEY = "posted_physical_source_authority"


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _utc(value, *, native=False):
    if not isinstance(value, datetime):
        raise ValueError("physical stream valid times must be datetimes")
    if value.tzinfo is None:
        if not native:
            raise ValueError("physical stream plans need explicit UTC offsets")
        # HorizontalSnapshot and the ordinary native real path use naive UTC.
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _trajectory(value):
    return {"source": value.source, "cycle": value.cycle.isoformat(),
            "member": value.member, "identity": value.identity}


def posted_source_identity(source_identity, *, input_plan):
    """Replace deferred digest leaves with explicit immutable plan references.

    A plan reference is a typed object, not a manifest digest. The ordinary
    source seal must still prove all its manifest rows against this exact
    plan and the actual posted markers before an ensemble seal is accepted.
    """
    from gpuwm.ingest.boundary_stream import INPUT_PLAN_SCHEMA, as_posted_placeholder, input_plan_sha256
    plan = json.loads(_canonical(input_plan))
    if (plan.get("schema") != INPUT_PLAN_SCHEMA or not isinstance(plan.get("manifest"), dict)
            or not _sha256(plan.get("route_table_sha256"))):
        raise ValueError("posted physical source needs the actual ordinary input-plan document")
    digest = input_plan_sha256(plan)
    placeholder = as_posted_placeholder(digest)
    paths = []
    def convert(value, path=()):
        if isinstance(value, dict):
            return {key: convert(item, (*path, key)) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item, (*path, index)) for index, item in enumerate(value)]
        if isinstance(value, str) and value.startswith("as-posted:"):
            if value != placeholder:
                raise ValueError("native source identity contains another posted input plan")
            paths.append(list(path))
            return {"schema": PLAN_REFERENCE_SCHEMA, "input_plan_sha256": digest,
                    "identity_path": list(path)}
        return value
    if SOURCE_AUTHORITY_KEY in source_identity:
        raise ValueError("native source identity is already bound to a posted physical plan")
    result = convert(source_identity)
    if ["input_manifest_sha256"] not in paths:
        raise ValueError("posted physical source must bind the ordinary deferred input manifest")
    # Manifest publication sorts mapping keys. Path order is an authority too,
    # so it must not depend on the source adapter's dictionary insertion order.
    paths.sort(key=_canonical)
    result[SOURCE_AUTHORITY_KEY] = {"schema": SOURCE_AUTHORITY_SCHEMA,
                                  "input_plan": plan, "input_plan_sha256": digest,
                                  "deferred_identity_paths": paths}
    return result


def _source_base(store_document):
    source = store_document["source"]
    return source["base"]["source"] if source.get("schema") == "gpuwm-ensemble-recentered-preparation.v1" else source


def _recipe_from_document(row):
    def source(value):
        return SourceTrajectory(value["source"], datetime.fromisoformat(value["cycle"]), value["member"])
    return SourceRecipe(row["kind"], source(row["base"]), datetime.fromisoformat(row["start"]),
                        datetime.fromisoformat(row["end"]),
                        tuple(RecipeMember(value["index"], value["seed"], source(value["trajectory"]))
                              for value in row["members"]),
                        tuple(source(value) for value in row["donor_population"]), row["calibration"],
                        row.get("perturbation"))


def validate_provider_plan(provider_plan):
    """Validate internal roster, source head and scientific contract authority."""
    if (not isinstance(provider_plan, dict)
            or set(provider_plan) != {"schema", "recipe", "recipe_sha256", "amplitude", "streams"}
            or provider_plan["schema"] != PROVIDER_PLAN_SCHEMA):
        raise ValueError("posted physical provider plan schema or shape differs")
    try:
        recipe = _recipe_from_document(provider_plan["recipe"])
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("posted physical provider plan has an incomplete source recipe") from error
    if recipe.describe() != provider_plan["recipe"] or recipe.sha256 != provider_plan["recipe_sha256"]:
        raise ValueError("posted physical provider recipe differs from its canonical identity")
    if recipe.kind not in RECIPE_KINDS or not recipe.members or _utc(recipe.end) <= _utc(recipe.start):
        raise ValueError("posted physical provider needs a known recipe and positive valid-time window")
    indices = tuple(member.index for member in recipe.members)
    if (any(type(index) is not int or index < 0 for index in indices) or len(set(indices)) != len(indices)
            or any(type(member.seed) is not int or not 0 <= member.seed < 1 << 64 for member in recipe.members)):
        raise ValueError("posted physical provider needs unique original member indices and unsigned seeds")
    identities = tuple(member.trajectory.identity for member in recipe.members)
    if recipe.kind == "control" and any(identity != recipe.base.identity for identity in identities):
        raise ValueError("every control member must retain the unchanged base trajectory")
    if recipe.kind == "surface-state":
        from gpuwm.ensemble.surface_controls import shared_surface_options
        shared_surface_options(recipe.perturbation, len(recipe.members))
        if any(identity != recipe.base.identity for identity in identities):
            raise ValueError("surface-state members must retain the unchanged base trajectory")
    if recipe.kind not in ("control", "surface-state") and len(set(identities)) != len(identities):
        raise ValueError("posted physical provider repeats a source member trajectory")
    if (isinstance(provider_plan["amplitude"], bool) or not isinstance(provider_plan["amplitude"], (int, float))
            or not math.isfinite(provider_plan["amplitude"]) or provider_plan["amplitude"] < 0):
        raise ValueError("posted physical provider needs a finite non-negative amplitude")
    required = {value.identity: value for value in recipe.acquisitions()}
    if set(provider_plan["streams"]) != set(required):
        raise ValueError("posted physical provider plan omits an original source acquisition")
    if recipe.kind == "recentered":
        donors = tuple(value.identity for value in recipe.donor_population)
        if len(donors) < 2 or len(set(donors)) != len(donors) or not set(identities) <= set(donors):
            raise ValueError("posted physical provider plan changed its fixed donor population")
    grids = set()
    from gpuwm.ensemble.physical_store import _validate_static_identity
    from gpuwm.ingest.boundary_stream import input_plan_sha256
    for identity, specification in provider_plan["streams"].items():
        if not isinstance(specification, dict) or set(specification) != {"head", "head_sha256"}:
            raise ValueError("posted provider source head has an unsupported shape")
        head = specification["head"]
        if (head.get("schema") != HEAD_SCHEMA
                or specification["head_sha256"] != hashlib.sha256((_canonical(head)+"\n").encode()).hexdigest()
                or head.get("trajectory") != _trajectory(required[identity])):
            raise ValueError("posted provider source head differs from its claimed hash or trajectory")
        times = tuple(_utc(datetime.fromisoformat(value)) for value in head["valid_times"])
        if (not times or any(left >= right for left, right in zip(times, times[1:]))
                or times[0] > recipe.start or times[-1] < recipe.end):
            raise ValueError("posted provider source plan does not cover the complete recipe window")
        if not _sha256(head.get("input_plan_sha256")):
            raise ValueError("posted provider source has no actual input-plan digest")
        validate_field_contract(head["field_contract"], head["grid"])
        _validate_static_identity(head["source"].get("static_identity"))
        authority = head["source"].get(SOURCE_AUTHORITY_KEY)
        if authority is not None and (authority.get("schema") != SOURCE_AUTHORITY_SCHEMA
                or input_plan_sha256(authority["input_plan"]) != head["input_plan_sha256"]
                or authority.get("input_plan_sha256") != head["input_plan_sha256"]):
            raise ValueError("posted physical source changed its typed input-plan authority")
        grids.add(_digest(head["grid"]))
    if len(grids) != 1:
        raise ValueError("posted provider source plans do not share native target geometry")
    return provider_plan


def validate_provider_receipt(receipt, *, provider_plan, member_index=None, valid_time=None):
    """Check one immutable member/time receipt against its frozen full plan."""
    validate_provider_plan(provider_plan)
    if (receipt.get("schema") != PROVIDER_SCHEMA
            or receipt.get("provider_plan_sha256") != _digest(provider_plan)
            or receipt.get("recipe_sha256") != provider_plan.get("recipe_sha256")
            or not _sha256(receipt.get("store_sha256"))):
        raise ValueError("posted physical receipt differs from its provider plan")
    members = {value["index"]: value for value in provider_plan["recipe"]["members"]}
    index = receipt.get("member_index")
    if index not in members or member_index is not None and index != member_index:
        raise ValueError("posted physical receipt renumbers the original recipe member")
    member = members[index]
    trajectory = SourceTrajectory(member["trajectory"]["source"],
                                  datetime.fromisoformat(member["trajectory"]["cycle"]),
                                  member["trajectory"]["member"])
    if receipt.get("member_seed") != member["seed"] or receipt.get("trajectory") != _trajectory(trajectory):
        raise ValueError("posted physical receipt changes the original member seed or trajectory")
    instant = _utc(datetime.fromisoformat(receipt["valid_time"]))
    if valid_time is not None and instant != _utc(valid_time, native=True):
        raise ValueError("posted physical receipt valid time differs from the requested native knot")
    source_plans = provider_plan["streams"]
    recipe = provider_plan["recipe"]
    if recipe["kind"] == "recentered":
        rows = [recipe["base"], *recipe["donor_population"]]
        required = {SourceTrajectory(row["source"], datetime.fromisoformat(row["cycle"]), row["member"]).identity
                    for row in rows}
    else:
        required = {trajectory.identity}
    if set(receipt.get("sources", {})) != required:
        raise ValueError("posted physical receipt omits a member of its fixed source population")
    for identity, frames in receipt["sources"].items():
        planned = source_plans[identity]
        times = tuple(datetime.fromisoformat(value) for value in planned["head"]["valid_times"])
        position = bisect_left(times, instant)
        expected = ((instant,) if position < len(times) and times[position] == instant
                    else times[position-1:position+1] if 0 < position < len(times) else ())
        if (not expected or len(frames) != len(expected)
                or tuple(datetime.fromisoformat(row["marker"]["valid_time"]) for row in frames) != expected):
            raise ValueError("posted physical receipt lacks its exact native time brackets")
        for frame in frames:
            marker = frame.get("marker", {})
            # Ready-marker files use the same canonical bytes plus newline.
            marker_hash = hashlib.sha256((_canonical(marker)+"\n").encode()).hexdigest()
            if (frame.get("head_sha256") != planned["head_sha256"]
                    or marker.get("head_sha256") != planned["head_sha256"]
                    or frame.get("marker_sha256") != marker_hash
                    or marker.get("schema") != FRAME_SCHEMA
                    or type(marker.get("index")) is not int
                    or marker["index"] != times.index(datetime.fromisoformat(marker["valid_time"]))
                    or not _sha256(marker.get("store_sha256"))):
                raise ValueError("posted physical receipt changes a consumed source marker")
            for evidence in (marker.get("posted_leads"), marker.get("decoded_leads")):
                if not isinstance(evidence, dict) or not evidence or any(not _sha256(value) for value in evidence.values()):
                    raise ValueError("posted physical receipt lacks actual source lead evidence")
    return receipt


def bind_posted_physical_input(store, receipt, *, provider_plan, member_index,
                               valid_time, grid, cfg, source_identity, input_plan,
                               static_identity):
    """Bind one member frame to actual ordinary native pre-initialization inputs."""
    from gpuwm.ensemble.physical_store import physical_input_binding
    validate_provider_receipt(receipt, provider_plan=provider_plan,
                              member_index=member_index, valid_time=valid_time)
    if (store.manifest_sha256 != receipt["store_sha256"] or len(store.times) != 1
            or _utc(store.times[0], native=True) != _utc(valid_time, native=True)):
        raise ValueError("posted physical member store differs from its provider receipt")
    expected = posted_source_identity(source_identity, input_plan=input_plan)
    binding = physical_input_binding(store, grid, cfg, expected,
                                      input_manifest_sha256=expected["input_manifest_sha256"],
                                      static_identity=static_identity)
    result = {"schema": INPUT_BINDING_SCHEMA, "provider_plan_sha256": _digest(provider_plan),
              "provider_receipt": receipt, "physical_input": binding}
    return validate_posted_physical_input(result, provider_plan=provider_plan,
                                          member_index=member_index, valid_time=valid_time)


def validate_posted_physical_input(binding, *, provider_plan, member_index=None, valid_time=None):
    """Validate a portable pre-real member binding without local source paths."""
    from gpuwm.ensemble.physical_store import validate_physical_input_binding, _semantic_source_identity
    if (binding.get("schema") != INPUT_BINDING_SCHEMA
            or binding.get("provider_plan_sha256") != _digest(provider_plan)):
        raise ValueError("posted native input binding differs from its provider plan")
    receipt = validate_provider_receipt(binding["provider_receipt"], provider_plan=provider_plan,
                                         member_index=member_index, valid_time=valid_time)
    physical = binding["physical_input"]
    if physical.get("manifest_sha256") != receipt["store_sha256"]:
        raise ValueError("posted native input manifest differs from its member receipt")
    document = physical["manifest"]
    base = _source_base(document)
    authority = base.get(SOURCE_AUTHORITY_KEY)
    if not isinstance(authority, dict) or authority.get("schema") != SOURCE_AUTHORITY_SCHEMA:
        raise ValueError("posted native input has no explicit immutable source-plan authority")
    from gpuwm.ingest.boundary_stream import as_posted_placeholder, input_plan_sha256
    plan_digest = input_plan_sha256(authority["input_plan"])
    if authority.get("input_plan_sha256") != plan_digest:
        raise ValueError("posted native source authority differs from its actual input plan")
    # Reconstruct and normalize the deferred source once more. This validates
    # every typed reference, its identity path and its plan, without treating
    # any plan digest as a completed raw input manifest.
    unbound = deepcopy(base)
    del unbound[SOURCE_AUTHORITY_KEY]
    for path in authority["deferred_identity_paths"]:
        owner = unbound
        for component in path[:-1]:
            owner = owner[component]
        expected = {"schema": PLAN_REFERENCE_SCHEMA, "input_plan_sha256": plan_digest,
                    "identity_path": path}
        if not path or owner[path[-1]] != expected:
            raise ValueError("posted native source has a malformed typed input-plan reference")
        owner[path[-1]] = as_posted_placeholder(plan_digest)
    if posted_source_identity(unbound, input_plan=authority["input_plan"]) != base:
        raise ValueError("posted native source-plan authority is not canonical")
    recipe = provider_plan["recipe"]
    selected = receipt["trajectory"]["identity"]
    if recipe["kind"] == "recentered":
        row = recipe["base"]
        base_identity = SourceTrajectory(row["source"], datetime.fromisoformat(row["cycle"]), row["member"]).identity
        expected_donors = {SourceTrajectory(row["source"], datetime.fromisoformat(row["cycle"]), row["member"]).identity
                           for row in recipe["donor_population"]}
        source = document["source"]
        if (source.get("schema") != "gpuwm-ensemble-recentered-preparation.v1"
                or set(source.get("donors", {})) != expected_donors
                or source.get("selected_member") != selected):
            raise ValueError("posted native input changed the frozen donor mean population")
        for identity, donor in source["donors"].items():
            if _semantic_source_identity(donor["source"]) != _semantic_source_identity(provider_plan["streams"][identity]["head"]["source"]):
                raise ValueError("posted native input donor differs from its captured source authority")
        if any(value.get("amplitude") != provider_plan["amplitude"] for value in source.get("bounds", {}).values()):
            raise ValueError("posted native input changed its declared recentering amplitude")
    else:
        base_identity = selected
    expected_base = provider_plan["streams"][base_identity]["head"]["source"]
    if _semantic_source_identity(base) != _semantic_source_identity(expected_base):
        raise ValueError("posted native input base differs from its captured source authority")
    validate_physical_input_binding(physical, input_manifest_sha256=base["input_manifest_sha256"],
                                      source_identity=expected_base,
                                      static_identity=expected_base["static_identity"])
    if (len(document["frames"]) != 1
            or _utc(datetime.fromisoformat(document["frames"][0]["valid_time"]), native=True)
            != _utc(datetime.fromisoformat(receipt["valid_time"]))):
        raise ValueError("posted native input binding is not exactly its requested physical knot")
    return binding


def validate_posted_native_source(binding, *, source_identity, input_manifest_authority,
                                  input_plan=None, static_identity=None):
    """Hold a member binding to an already verified ordinary native source.

    At a head, ``input_plan`` must be the ordinary head's exact plan. At a
    seal the caller first performs the ordinary manifest/source checks; only
    the captured deferred digest paths may then return to typed plan refs for
    comparison. All scientific source controls and actual statics still agree.
    """
    from gpuwm.ensemble.physical_store import _semantic_source_identity, _validate_static_identity
    from gpuwm.ingest.boundary_stream import as_posted_placeholder, input_plan_sha256
    if binding.get("schema") != INPUT_BINDING_SCHEMA:
        raise ValueError("posted native source needs its full physical member binding")
    captured = _source_base(binding["physical_input"]["manifest"])
    authority = captured.get(SOURCE_AUTHORITY_KEY)
    if not isinstance(authority, dict) or authority.get("schema") != SOURCE_AUTHORITY_SCHEMA:
        raise ValueError("posted native source lacks its immutable plan authority")
    plan_digest = input_plan_sha256(authority["input_plan"])
    if authority.get("input_plan_sha256") != plan_digest:
        raise ValueError("posted native source-plan authority has changed")
    if input_plan is not None and (input_plan_sha256(input_plan) != plan_digest or input_plan != authority["input_plan"]):
        raise ValueError("ordinary native source and posted physical member have different input plans")
    expected_placeholder = as_posted_placeholder(plan_digest)
    if input_manifest_authority == expected_placeholder:
        if input_plan is None:
            raise ValueError("posted native source head must verify its actual input plan")
        sealed = False
    elif _sha256(input_manifest_authority):
        sealed = True
    else:
        raise ValueError("ordinary native source has no verified manifest or exact posted-plan authority")
    actual = deepcopy(source_identity)
    if actual.get("input_manifest_sha256") != input_manifest_authority:
        raise ValueError("ordinary native source identity differs from its verified input manifest authority")
    for path in authority["deferred_identity_paths"]:
        owner = actual
        wanted = captured
        try:
            for component in path[:-1]:
                owner, wanted = owner[component], wanted[component]
            value = owner[path[-1]]
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("ordinary native source dropped a captured deferred digest path") from error
        if (sealed and not _sha256(value)) or (not sealed and value != expected_placeholder):
            raise ValueError("ordinary native source deferred field is not its verified digest authority")
        owner[path[-1]] = deepcopy(wanted[path[-1]])
    actual[SOURCE_AUTHORITY_KEY] = deepcopy(authority)
    if _semantic_source_identity(actual) != _semantic_source_identity(captured):
        raise ValueError("posted physical input scientific source authority differs from ordinary native preparation")
    if static_identity is not None and _validate_static_identity(captured.get("static_identity")) != _validate_static_identity(static_identity):
        raise ValueError("posted physical input statics differ from ordinary native preparation")
    return binding


def _safe_artifact(root, name):
    name = str(name)
    if "\\" in name or ":" in name or any(part in ("", ".", "..") for part in name.split("/")):
        raise ValueError("posted source capsule artifact name is not a safe relative path")
    path = Path(root) / name
    if Path(name).is_absolute() or not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("posted source capsule artifact escapes its own root")
    return path


def _source_artifacts(head):
    from gpuwm.ingest.boundary_stream import proof_document_name
    posted = head["basis"].get("as_posted")
    if not posted or head["basis"].get("tree") is not None:
        raise ValueError("posted physical sources require an ordinary single-domain posted source head")
    names = {"boundary-stream/head.json", "boundary-stream/posted-leads.json",
             str(head["basis"]["cache"]["directory"])+"/header.json",
             proof_document_name(head), posted["manifest_path"]}
    names.update(f"boundary-stream/segments/{index:05d}.json"
                 for index in range(len(head["basis"]["cache"]["lbc"]["schedule"])))
    names.update(value["path"] for value in posted.get("documents", {}).values())
    return names


def capture_source_seal(prepared_root, stream, *, expected_head_sha256=None):
    """Capture portable metadata after ordinary source seal verification.

    The capsule is replayed through the ordinary verifier by downstream
    readers. It carries actual input-manifest bytes and all posted markers,
    document row authorities and segment records required by that verifier.
    It does not replace raw-content verification with a receipt assertion.
    """
    from gpuwm.ingest.boundary_stream import read_head, verify_seal
    prepared_root = Path(prepared_root)
    head = read_head(prepared_root, expected_sha256=expected_head_sha256)
    result = verify_seal(prepared_root, head=head)
    if head["basis"]["as_posted"]["input_plan_sha256"] != stream.head["input_plan_sha256"]:
        raise ValueError("ordinary source seal belongs to another physical input plan")
    artifacts = {name: _safe_artifact(prepared_root, name).read_bytes().decode("utf-8")
                 for name in sorted(_source_artifacts(head))}
    frames = [stream.require(value)[1] for value in stream.times]
    certificate = {"schema": SOURCE_SEAL_SCHEMA, "physical_head_sha256": stream.head_sha256,
                   "input_plan_sha256": stream.head["input_plan_sha256"],
                   "ordinary_seal": result, "artifacts": artifacts, "physical_frames": frames}
    validate_source_seal(certificate, physical_head={"head_sha256": stream.head_sha256, "head": stream.head})
    return certificate


def validate_source_seal(certificate, *, physical_head):
    """Replay one complete ordinary source seal and hold physical markers to it."""
    from gpuwm.ingest.boundary_stream import (
        decoded_lead_record_sha256, read_head, verify_seal,
    )
    if (certificate.get("schema") != SOURCE_SEAL_SCHEMA
            or certificate.get("physical_head_sha256") != physical_head["head_sha256"]
            or certificate.get("input_plan_sha256") != physical_head["head"]["input_plan_sha256"]):
        raise ValueError("posted source seal differs from its pinned physical plan")
    artifacts = certificate.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("posted source seal has no portable ordinary-source artifacts")
    with tempfile.TemporaryDirectory(prefix="gpuwm-posted-source-") as temporary:
        root = Path(temporary)
        for name, content in artifacts.items():
            path = _safe_artifact(root, name)
            if not isinstance(content, str):
                raise ValueError("posted source capsule metadata must be UTF-8 text")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content.encode("utf-8"))
        head = read_head(root)
        if set(artifacts) != _source_artifacts(head):
            raise ValueError("posted source capsule is not the exact ordinary seal artifact set")
        result = verify_seal(root, head=head)
        if result != certificate["ordinary_seal"]:
            raise ValueError("posted source capsule differs from its ordinary verified seal")
        posted = head["basis"]["as_posted"]
        if posted["input_plan_sha256"] != certificate["input_plan_sha256"]:
            raise ValueError("ordinary sealed source differs from the physical input plan")
        record = json.loads(artifacts["boundary-stream/posted-leads.json"])
    captured = physical_head["head"]["source"]
    if SOURCE_AUTHORITY_KEY in captured:
        from gpuwm.ensemble.physical_store import _semantic_source_identity
        from gpuwm.ingest.boundary_stream import as_posted_placeholder
        # Some native adapters bind the raw manifest only in the outer cache
        # identity. Ordinary seal verification above already holds that outer
        # authority to the plan; preserve any explicit inner value/conflict.
        native_identity = {"input_manifest_sha256": as_posted_placeholder(posted["input_plan_sha256"]),
                           **head["basis"]["cache"]["identity"]["source_identity"]}
        ordinary = posted_source_identity(native_identity,
                                          input_plan=posted["input_plan"])
        if _semantic_source_identity(captured) != _semantic_source_identity(ordinary):
            raise ValueError("physical captured source differs from its ordinary prepared source authority")
    frames = certificate.get("physical_frames", [])
    if [value.get("marker", {}).get("valid_time") for value in frames] != physical_head["head"]["valid_times"]:
        raise ValueError("posted source seal does not cover every planned physical valid time")
    for frame in frames:
        marker = frame["marker"]
        expected_hash = hashlib.sha256((_canonical(marker)+"\n").encode()).hexdigest()
        if (frame.get("head_sha256") != physical_head["head_sha256"]
                or marker.get("head_sha256") != physical_head["head_sha256"]
                or frame.get("marker_sha256") != expected_hash):
            raise ValueError("posted source seal changes a physical ready marker")
        for lead, digest in marker["posted_leads"].items():
            row = record["leads"].get(lead)
            if row is None or row.get("marker_sha256") != digest:
                raise ValueError("physical frame consumed a raw lead absent from the ordinary source seal")
        # Native routes with decoded-row documents already prove these rows
        # against the final source document through the ordinary verifier.
        for lead, digest in marker["decoded_leads"].items():
            decoded = record["leads"].get(lead, {}).get("decoded")
            if decoded is not None and decoded_lead_record_sha256(decoded) != digest:
                raise ValueError("physical decoded lead differs from the ordinary sealed source rows")
    return certificate


def validate_provider_seal(seal, *, provider_plan, receipts=()):
    """Verify every source completion and each consumed original member frame."""
    validate_provider_plan(provider_plan)
    if (seal.get("schema") != PROVIDER_SEAL_SCHEMA
            or seal.get("provider_plan_sha256") != _digest(provider_plan)
            or set(seal.get("sources", {})) != set(provider_plan["streams"])):
        raise ValueError("posted provider seal changes its fixed source population")
    source_frames = {}
    for identity, certificate in seal["sources"].items():
        validate_source_seal(certificate, physical_head=provider_plan["streams"][identity])
        source_frames[identity] = {value["marker"]["valid_time"]: value for value in certificate["physical_frames"]}
    for receipt in receipts:
        validate_provider_receipt(receipt, provider_plan=provider_plan)
        for identity, frames in receipt["sources"].items():
            for frame in frames:
                if source_frames[identity].get(frame["marker"]["valid_time"]) != frame:
                    raise ValueError("posted provider seal changed a source frame consumed by a member")
    return seal


def _publish_json(path, document):
    """Publish a complete JSON file without ever replacing a ready marker."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / ("." + path.name + "." + uuid.uuid4().hex + ".partial")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(_canonical(document) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _link_or_copy(source, target):
    try:
        os.link(source, target)
    except OSError:
        # Different filesystems cannot hard-link. Exclusive creation also
        # makes the copy fallback refuse any existing destination.
        with Path(source).open("rb") as reader, Path(target).open("xb") as writer:
            shutil.copyfileobj(reader, writer)


class PhysicalFramePending(RuntimeError):
    """The requested source frame is planned but has not been published."""
    def __init__(self, trajectory, valid_time):
        self.trajectory = trajectory
        self.valid_time = valid_time
        super().__init__(f"physical source {trajectory.source} {trajectory.identity} "
                         f"has not published valid time {valid_time.isoformat()}")


class PostedPhysicalStream:
    """One source-neutral trajectory with independently ready native frames.

    ``source_identity`` is the ordinary native source identity at its posted
    head, including its captured ``static_identity``. ``input_plan_sha256``
    is the ordinary as-posted input-plan digest. A producer must verify raw
    posted objects and decoded rows through its existing source adapter
    before passing their receipt hashes to :meth:`publish`.
    """
    def __init__(self, root, *, wait=None):
        self.root = Path(root).resolve()
        self.head_path = self.root / "physical-head.json"
        raw = self.head_path.read_bytes()
        self.head_sha256 = hashlib.sha256(raw).hexdigest()
        self.head = json.loads(raw)
        self._head_document_sha256 = _digest(self.head)
        if self.head.get("schema") != HEAD_SCHEMA:
            raise ValueError("physical stream head schema differs")
        row = self.head["trajectory"]
        self.trajectory = SourceTrajectory(row["source"], datetime.fromisoformat(row["cycle"]), row["member"])
        if row != _trajectory(self.trajectory):
            raise ValueError("physical stream trajectory differs from its canonical source identity")
        self.times = tuple(_utc(datetime.fromisoformat(value)) for value in self.head["valid_times"])
        if not self.times or any(a >= b for a, b in zip(self.times, self.times[1:])):
            raise ValueError("physical stream plan needs strictly increasing valid times")
        if not _sha256(self.head.get("input_plan_sha256")):
            raise ValueError("physical stream needs a verified input-plan digest")
        validate_field_contract(self.head.get("field_contract"), self.head["grid"])
        from gpuwm.ensemble.physical_store import _validate_static_identity
        _validate_static_identity(self.head["source"].get("static_identity"))
        self.wait = wait
        self._consumed = {}

    @classmethod
    def create(cls, root, *, trajectory, valid_times, grid_identity,
               source_identity, field_contract, input_plan_sha256, wait=None):
        if not isinstance(trajectory, SourceTrajectory):
            raise ValueError("physical stream needs a canonical source trajectory")
        times = tuple(_utc(value) for value in valid_times)
        if not times or any(a >= b for a, b in zip(times, times[1:])):
            raise ValueError("physical stream plan needs strictly increasing valid times")
        if not _sha256(input_plan_sha256):
            raise ValueError("physical stream needs a verified input-plan digest")
        validate_field_contract(field_contract, grid_identity)
        from gpuwm.ensemble.physical_store import _validate_static_identity
        _validate_static_identity(source_identity.get("static_identity"))
        _publish_json(Path(root) / "physical-head.json", {
            "schema": HEAD_SCHEMA, "trajectory": _trajectory(trajectory),
            "valid_times": [value.isoformat() for value in times],
            "grid": grid_identity, "source": source_identity,
            "field_contract": field_contract, "input_plan_sha256": input_plan_sha256})
        return cls(root, wait=wait)

    def _check_head(self):
        if (digest_file(self.head_path) != self.head_sha256
                or _digest(self.head) != self._head_document_sha256):
            raise ValueError("physical stream head changed after its authority was loaded")

    def _index(self, valid_time):
        value = _utc(valid_time)
        index = bisect_left(self.times, value)
        if index == len(self.times) or self.times[index] != value:
            raise ValueError("requested physical valid time is absent from the source plan")
        return index

    def _marker(self, index):
        return self.root / "ready" / f"{index:05d}.json"

    def publish(self, snapshot, *, posted_leads, decoded_leads):
        """Publish one native frame after its source adapter verifies evidence.

        Evidence maps use stable lead/object labels and SHA-256 values. They
        are independent of future leads; neither a complete input manifest
        nor a complete physical trajectory is needed to publish this frame.
        """
        self._check_head()
        index = self._index(_utc(snapshot.valid_time, native=True))
        if (self.root / "physical-seal.json").exists() or self._marker(index).exists():
            raise FileExistsError("a published physical frame or stream is immutable")
        for evidence in (posted_leads, decoded_leads):
            if (not isinstance(evidence, dict) or not evidence
                    or any(not isinstance(key, str) or not key or not _sha256(value)
                           for key, value in evidence.items())):
                raise ValueError("physical frame needs posted and decoded lead digest maps")
        frame_root = self.root / "frames" / f"{index:05d}"
        frame_root.mkdir(parents=True, exist_ok=False)
        store = NativePhysicalStore(frame_root, grid_identity=self.head["grid"],
                                    source_identity=self.head["source"],
                                    field_contract=self.head["field_contract"])
        store.write(replace(snapshot, valid_time=_utc(snapshot.valid_time, native=True).replace(tzinfo=None)))
        store.seal()
        marker = {"schema": FRAME_SCHEMA, "head_sha256": self.head_sha256,
                  "index": index, "valid_time": self.times[index].isoformat(),
                  "store_sha256": store.manifest_sha256,
                  "posted_leads": dict(posted_leads), "decoded_leads": dict(decoded_leads)}
        _publish_json(self._marker(index), marker)
        return marker

    def require(self, valid_time):
        """Return a verified one-frame store, waiting only through the caller."""
        self._check_head()
        index = self._index(valid_time)
        path = self._marker(index)
        if not path.exists() and self.wait is not None:
            self.wait(self.trajectory, self.times[index])
            self._check_head()
        if not path.exists():
            raise PhysicalFramePending(self.trajectory, self.times[index])
        raw = path.read_bytes()
        marker_sha256 = hashlib.sha256(raw).hexdigest()
        if index in self._consumed and self._consumed[index] != marker_sha256:
            raise ValueError("physical frame marker changed after it was consumed")
        marker = json.loads(raw)
        if (marker.get("schema") != FRAME_SCHEMA or marker.get("head_sha256") != self.head_sha256
                or marker.get("index") != index or marker.get("valid_time") != self.times[index].isoformat()):
            raise ValueError("physical frame marker differs from its source plan")
        for key in ("posted_leads", "decoded_leads"):
            evidence = marker.get(key)
            if (not isinstance(evidence, dict) or not evidence
                    or any(not isinstance(name, str) or not name or not _sha256(value)
                           for name, value in evidence.items())):
                raise ValueError("physical frame marker lacks source evidence")
        store = NativePhysicalStore(self.root / "frames" / f"{index:05d}")
        if store.manifest_sha256 != marker.get("store_sha256"):
            raise ValueError("physical frame store differs from its ready marker")
        if (store.document["grid"] != self.head["grid"] or store.document["source"] != self.head["source"]
                or store.field_contract != self.head["field_contract"]
                or tuple(_utc(time, native=True) for time in store.times) != (self.times[index],)):
            raise ValueError("physical frame authority differs from its source plan")
        store.read(0)  # Validate native file headers, units, arrays and payload digest.
        self._consumed[index] = marker_sha256
        return store, {"head_sha256": self.head_sha256, "marker_sha256": marker_sha256,
                       "marker": marker}

    def bracket_times(self, valid_time):
        """Exact knot or two planned brackets; never extrapolate a trajectory."""
        value = _utc(valid_time)
        position = bisect_left(self.times, value)
        if position < len(self.times) and self.times[position] == value:
            return (value,)
        if position == 0 or position == len(self.times):
            raise ValueError("physical source plan does not bracket the requested valid time")
        return self.times[position-1:position+1]

    def seal(self):
        """Bind the complete ready sequence without changing any consumed frame."""
        receipts = [self.require(value)[1] for value in self.times]
        document = {"schema": SEAL_SCHEMA, "head_sha256": self.head_sha256,
                    "frames": [item["marker_sha256"] for item in receipts]}
        _publish_json(self.root / "physical-seal.json", document)
        return document


@dataclass(frozen=True)
class PostedSourceContext:
    """Pinned ordinary source preparation reused by one native member route.

    Geometry and surface arrays remain owned by the ordinary source's checked
    head. Native adapters use their normal source preflight before reading
    those arrays. The methods below keep the original producer wait/failure
    and final source seal checks, including when the source is already sealed.
    """
    trajectory: SourceTrajectory
    prepared_root: Path
    prepared_head: dict
    physical_stream: PostedPhysicalStream
    source_plan: dict
    _head_pin: str
    _on_wait: object = None

    def verify(self):
        from gpuwm.ingest.boundary_stream import read_head
        self.physical_stream._check_head()
        head = read_head(self.prepared_root, expected_sha256=self._head_pin)
        posted = head["basis"].get("as_posted") or {}
        if (head != self.prepared_head or posted.get("input_plan") != self.source_plan
                or posted.get("input_plan_sha256") != self.physical_stream.head["input_plan_sha256"]):
            raise ValueError("ordinary source context changed after its captured physical head was bound")
        return self

    def intervals(self):
        from gpuwm.ingest.boundary_stream import StreamedIntervals
        self.verify()
        return StreamedIntervals(self.prepared_root, head=self.prepared_head, on_wait=self._on_wait)

    def require_interval(self, index):
        """Use the ordinary source's existing ready-marker and producer wait."""
        result = self.intervals().require(index)
        self.verify()
        return result

    def wait_sealed(self):
        self.intervals().wait_sealed()
        self.verify()

    def capture_seal(self):
        """Wait for all raw inputs and retain the exact verified source seal."""
        self.wait_sealed()
        result = capture_source_seal(self.prepared_root, self.physical_stream,
                                     expected_head_sha256=self._head_pin)
        self.verify()
        return result


def _window(root, stores):
    """Build a small immutable catalog over verified native frame bytes."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    first = stores[0]
    document = deepcopy(first.document)
    document["frames"] = []
    for index, store in enumerate(stores):
        if (store.document["grid"] != first.document["grid"]
                or store.document["source"] != first.document["source"]
                or store.field_contract != first.field_contract):
            raise ValueError("physical window changes its source, units or grid authority")
        store.read(0)
        frame = deepcopy(store.document["frames"][0])
        filename = f"frame-{index:04d}.nc"
        _link_or_copy(store.root / frame["file"], root / filename)
        frame["file"] = filename
        document["frames"].append(frame)
    _publish_json(root / "physical-store.json", document)
    return root


class PostedPhysicalProvider:
    """Resolve original recipe members at each initial or boundary valid time.

    Every recentered request resolves the full frozen donor population, even
    for a one-member replay. Native pressure and time interpolation, humidity
    conversion and recentering are delegated to RecenteredPhysicalPreparation.
    Output keys are original RecipeMember indices, never local batch offsets.
    """
    def __init__(self, recipe, streams, *, amplitude=1.0, cpu_bridge=None, workers=1):
        if not isinstance(recipe, SourceRecipe):
            raise ValueError("posted provider needs a resolved source recipe")
        required = {item.identity: item for item in recipe.acquisitions()}
        if set(streams) != set(required):
            raise ValueError("posted provider streams must cover the complete recipe acquisitions")
        self.streams = dict(streams)
        self.recipe = recipe
        self.amplitude, self.cpu_bridge, self.workers = amplitude, cpu_bridge, workers
        self.root = None
        self.prepared_roots = {}
        self.prepared_headpins = {}
        self.on_wait = None
        for identity, stream in self.streams.items():
            if not isinstance(stream, PostedPhysicalStream) or stream.trajectory != required[identity]:
                raise ValueError("posted provider source identity differs from its recipe trajectory")
        grids = {_digest(value.head["grid"]) for value in self.streams.values()}
        if len(grids) != 1:
            raise ValueError("posted physical trajectories must use the same native target geometry")
        if recipe.kind == "recentered":
            population = tuple(value.identity for value in recipe.donor_population)
            if len(population) < 2 or len(set(population)) != len(population):
                raise ValueError("posted recentering needs a fixed distinct donor population")
            if any(value.trajectory.identity not in population for value in recipe.members):
                raise ValueError("posted output member is outside the frozen donor population")
        self.plan = {"schema": PROVIDER_PLAN_SCHEMA, "recipe": recipe.describe(),
                     "recipe_sha256": recipe.sha256, "amplitude": amplitude,
                     "streams": {key: {"head_sha256": value.head_sha256, "head": deepcopy(value.head)}
                                 for key, value in sorted(self.streams.items())}}
        validate_provider_plan(self.plan)

    def write_plan(self, root, *, prepared_roots):
        """Publish a provider usable by native preparer child processes.

        Ordinary source prepared heads must already exist. Their initial
        physical frame precedes that publication. Subsequent frame waits use
        the same source's ordinary StreamedIntervals producer lifecycle.
        """
        if set(prepared_roots) != set(self.streams):
            raise ValueError("serialized posted provider needs every ordinary source prepared root")
        from gpuwm.ingest.boundary_stream import read_head
        records = {}
        for identity, path in prepared_roots.items():
            head = read_head(path)
            if (head["basis"].get("as_posted") or {}).get("input_plan_sha256") != self.streams[identity].head["input_plan_sha256"]:
                raise ValueError("ordinary source prepared head differs from the physical input plan")
            self.streams[identity].require(self.streams[identity].times[0])
            records[identity] = {"stream_root": str(self.streams[identity].root),
                                 "prepared_root": str(Path(path).resolve()),
                                 "prepared_head_sha256": head["head_sha256"]}
        self.root = Path(root).resolve()
        _publish_json(self.root / "provider-head.json", {"plan": self.plan, "sources": records})
        self.prepared_roots = {key: Path(value).resolve() for key, value in prepared_roots.items()}
        self.prepared_headpins = {key: value["prepared_head_sha256"] for key, value in records.items()}
        self._bind_source_waits()
        return self.root

    @classmethod
    def open(cls, root, *, cpu_bridge=None, workers=1, on_wait=None):
        """Open the same frozen member plan in another native prep process."""
        root = Path(root).resolve()
        document = json.loads((root/"provider-head.json").read_bytes())
        plan = document["plan"]
        validate_provider_plan(plan)
        recipe = _recipe_from_document(plan["recipe"])
        from gpuwm.ingest.boundary_stream import read_head
        streams, prepared_roots, prepared_headpins = {}, {}, {}
        for identity, specification in document["sources"].items():
            prepared = Path(specification["prepared_root"]).resolve()
            prepared_head = read_head(prepared, expected_sha256=specification["prepared_head_sha256"])
            stream = PostedPhysicalStream(specification["stream_root"])
            if (prepared_head["basis"].get("as_posted") or {}).get("input_plan_sha256") != stream.head["input_plan_sha256"]:
                raise ValueError("serialized physical provider changed its ordinary source input plan")
            streams[identity], prepared_roots[identity] = stream, prepared
            prepared_headpins[identity] = specification["prepared_head_sha256"]
        result = cls(recipe, streams, amplitude=plan["amplitude"], cpu_bridge=cpu_bridge, workers=workers)
        if result.plan != plan:
            raise ValueError("serialized posted provider changed its recipe or pinned source heads")
        result.root, result.prepared_roots = root, prepared_roots
        result.prepared_headpins = prepared_headpins
        result.set_wait_observer(on_wait)
        result._bind_source_waits()
        return result

    def set_wait_observer(self, observer):
        """Attach this consumer's observer, including to existing contexts.

        A native member creates its writer after opening source heads. The
        dynamic dispatch lets that writer observe later source waits without
        forwarding its stop request to the shared ordinary producer.
        """
        if observer is not None and not callable(observer):
            raise TypeError("posted source wait observer must be callable or None")
        self.on_wait = observer

    def _notify_wait(self, report):
        if self.on_wait is not None:
            self.on_wait(report)

    def _bind_source_waits(self):
        from gpuwm.ingest.boundary_stream import StreamedIntervals, read_head
        for identity, stream in self.streams.items():
            prepared, pin = self.prepared_roots[identity], self.prepared_headpins[identity]
            head = read_head(prepared, expected_sha256=pin)
            intervals = StreamedIntervals(prepared, head=head, on_wait=self._notify_wait)

            def wait(trajectory, valid_time, *, stream=stream, intervals=intervals,
                     prepared=prepared, pin=pin):
                read_head(prepared, expected_sha256=pin)
                position = stream._index(valid_time)
                if position == 0:
                    raise ValueError("ordinary source head has no initial physical frame")
                intervals.require(position-1)
                read_head(prepared, expected_sha256=pin)

            stream.wait = wait

    def _source_context(self, identity):
        from gpuwm.ingest.boundary_stream import read_head
        if (self.root is None or identity not in self.prepared_roots
                or identity not in self.prepared_headpins):
            raise ValueError("posted source context requires a serialized original prepared head")
        stream = self.streams[identity]
        stream._check_head()
        head = read_head(self.prepared_roots[identity], expected_sha256=self.prepared_headpins[identity])
        posted = head["basis"].get("as_posted") or {}
        if posted.get("input_plan_sha256") != stream.head["input_plan_sha256"]:
            raise ValueError("ordinary source context differs from its captured physical input plan")
        context = PostedSourceContext(stream.trajectory, self.prepared_roots[identity], deepcopy(head),
                                      stream, deepcopy(posted["input_plan"]),
                                      self.prepared_headpins[identity], self._notify_wait)
        return context.verify()

    def source_context(self, member_index):
        """The original checked source head shared by a native member route."""
        members = {value.index: value for value in self.recipe.members}
        if type(member_index) is not int or member_index not in members:
            raise ValueError("posted source context needs an original RecipeMember index")
        trajectory = self.recipe.base if self.recipe.kind == "recentered" else members[member_index].trajectory
        return self._source_context(trajectory.identity)

    def resolve(self, member_index, valid_time):
        """Return a checked one-frame store and receipt for a native real call."""
        if self.root is None:
            raise ValueError("posted provider resolve requires a published provider root")
        instant = _utc(valid_time, native=True)
        known = {item.index for item in self.recipe.members}
        if type(member_index) is not int or member_index not in known:
            raise ValueError("posted resolver needs an original RecipeMember index")
        output = self.root / "members" / str(member_index) / instant.strftime("%Y%m%dT%H%M%SZ")
        receipt_path = output / "posted-physical-receipt.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_bytes())
            validate_provider_receipt(receipt, provider_plan=self.plan,
                                      member_index=member_index, valid_time=instant)
            for identity, frames in receipt["sources"].items():
                for frame in frames:
                    current = self.streams[identity].require(datetime.fromisoformat(frame["marker"]["valid_time"]))[1]
                    if current != frame:
                        raise ValueError("cached posted member differs from its consumed source markers")
        else:
            result = self.prepare(instant, {member_index: output},
                                  work_root=self.root/"work"/uuid.uuid4().hex)
            receipt = result[member_index]["receipt"]
        store = NativePhysicalStore(output)
        if store.manifest_sha256 != receipt["store_sha256"]:
            raise ValueError("resolved posted member differs from its immutable receipt")
        store.read(0)
        return store, receipt

    def seal(self):
        """Wait for and verify every ordinary source's final raw-input seal."""
        if self.root is None or set(self.prepared_roots) != set(self.streams):
            raise ValueError("posted provider seal requires its serialized ordinary source roots")
        certificates = {}
        for identity, stream in sorted(self.streams.items()):
            certificates[identity] = self._source_context(identity).capture_seal()
        result = {"schema": PROVIDER_SEAL_SCHEMA, "provider_plan_sha256": _digest(self.plan),
                  "sources": certificates}
        validate_provider_seal(result, provider_plan=self.plan)
        target = self.root/"provider-seal.json"
        if target.exists():
            if json.loads(target.read_bytes()) != result:
                raise ValueError("posted provider final source seal changed after publication")
        else:
            try:
                _publish_json(target, result)
            except FileExistsError:
                if json.loads(target.read_bytes()) != result:
                    raise ValueError("concurrent posted provider seals disagree") from None
        return result

    def prepare(self, valid_time, outputs, *, work_root):
        """Publish selected member stores for one time, without a future seal.

        ``work_root`` must be a new owned directory. For a planned but absent
        frame the source wait/failure propagates before any output is written.
        Caller retry after a wait uses a fresh work directory.
        """
        instant = _utc(valid_time)
        if not self.recipe.start <= instant <= self.recipe.end:
            raise ValueError("posted member valid time falls outside the recipe window")
        known = {item.index: item for item in self.recipe.members}
        if (not outputs or any(type(key) is not int or key not in known for key in outputs)
                or len({Path(value).resolve() for value in outputs.values()}) != len(outputs)):
            raise ValueError("posted output paths must select distinct original member indices")
        needed = ({self.recipe.base.identity: (instant,)} if self.recipe.kind == "recentered" else {})
        trajectories = (self.recipe.donor_population if self.recipe.kind == "recentered"
                        else tuple(known[index].trajectory for index in outputs))
        for trajectory in trajectories:
            stream = self.streams[trajectory.identity]
            needed[trajectory.identity] = stream.bracket_times(instant)
        if self.recipe.kind != "recentered" and any(len(times) != 1 for times in needed.values()):
            raise ValueError("direct posted recipes require exact native source knots")
        # Resolve every required donor first: a late final donor must not leave
        # a partial member population or a changed ensemble mean on disk.
        resolved = {key: [self.streams[key].require(value) for value in times]
                    for key, times in sorted(needed.items())}
        scratch = Path(work_root)
        scratch.mkdir(parents=True, exist_ok=False)
        windows = {key: _window(scratch / key, [item[0] for item in values])
                   for key, values in resolved.items()}
        if self.recipe.kind == "recentered":
            from gpuwm.ensemble.physical_recenter import RecenteredPhysicalPreparation
            operator = RecenteredPhysicalPreparation(
                windows[self.recipe.base.identity],
                {value.identity: windows[value.identity] for value in self.recipe.donor_population},
                amplitude=self.amplitude, cpu_bridge=self.cpu_bridge, workers=self.workers)
            selected = {known[index].trajectory.identity: path for index, path in outputs.items()}
            if len(selected) != len(outputs):
                raise ValueError("posted recipe repeats a selected donor trajectory")
            operator.prepare(selected)
        else:
            for index, path in outputs.items():
                _window(path, [resolved[known[index].trajectory.identity][0][0]])
        receipts = {}
        for index, path in outputs.items():
            member = known[index]
            store = NativePhysicalStore(path)
            receipt = {"schema": PROVIDER_SCHEMA, "recipe_sha256": self.recipe.sha256,
                       "provider_plan_sha256": _digest(self.plan),
                       "member_index": member.index, "member_seed": member.seed,
                       "trajectory": _trajectory(member.trajectory),
                       "valid_time": instant.isoformat(), "store_sha256": store.manifest_sha256,
                       "sources": {key: [item[1] for item in values] for key, values in resolved.items()
                                   if self.recipe.kind == "recentered" or key == member.trajectory.identity}}
            validate_provider_receipt(receipt, provider_plan=self.plan, member_index=index, valid_time=instant)
            _publish_json(Path(path) / "posted-physical-receipt.json", receipt)
            receipts[index] = {"path": str(store.manifest_path), "sha256": store.manifest_sha256,
                               "receipt": receipt}
        return receipts
