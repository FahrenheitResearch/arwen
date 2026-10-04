"""Posted physical frames preserve the sealed native numerical contract."""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import os

import numpy as np
import pytest

from gpuwm.ensemble.physical_store import NativePhysicalStore, physical_static_identity
from gpuwm.ensemble.posted_physical import (
    PhysicalFramePending, PostedPhysicalProvider, PostedPhysicalStream,
)
from gpuwm.ensemble.recipes import RecipeMember, SourceRecipe, SourceTrajectory, ensemble_population
from test_ensemble_physical_preparation import snapshot
from physical_field_fixtures import analytic_field_contract

START = datetime(2024, 1, 1, tzinfo=timezone.utc)
GRID = {"mass_shape": [2, 3], "fixture": "native-field-test"}


@pytest.fixture
def bridge():
    value = os.environ.get("GPUWM_ENSEMBLE_PREPARATION_BRIDGE")
    if not value:
        pytest.skip("requires the native preparation and NetCDF artifacts")
    return value


def recipe():
    population = ensemble_population("gefs", START, perturbed_only=True)[:3]
    return SourceRecipe("recentered", SourceTrajectory("gfs", START), START,
                        START + timedelta(hours=2),
                        tuple(RecipeMember(index, 1000+index, source)
                              for index, source in zip((7, 12, 29), population)), population)


def make_stream(root, trajectory, hours, *, wait=None, contract=None):
    return PostedPhysicalStream.create(
        root, trajectory=trajectory, valid_times=[START+timedelta(hours=hour) for hour in hours],
        grid_identity=GRID, source_identity={
            "input_manifest_sha256": "a"*64, "source_trajectory": trajectory.identity,
            "static_identity": physical_static_identity({"HGT_M": np.zeros((2, 3))})},
        field_contract=analytic_field_contract(GRID) if contract is None else contract,
        input_plan_sha256="b"*64, wait=wait)


def publish(stream, hour, offset=0.0):
    return stream.publish(snapshot(hour, offset), posted_leads={str(hour): "c"*64},
                          decoded_leads={str(hour): "d"*64})


def streams(root, plan):
    return {item.identity: make_stream(root/item.identity, item,
                                       (0, 1, 2) if item == plan.base else (0, 3))
            for item in plan.acquisitions()}


def sealed_store(root, stream, frames):
    store = NativePhysicalStore(root, grid_identity=stream.head["grid"],
                                source_identity=stream.head["source"],
                                field_contract=stream.head["field_contract"])
    for frame in frames:
        store.write(frame)
    store.seal()
    return root


def assert_same_frame(first, second):
    assert first.valid_time == second.valid_time
    assert first.levels_hpa.tobytes() == second.levels_hpa.tobytes()
    assert first.fields.keys() == second.fields.keys()
    assert all(first.fields[key].tobytes() == second.fields[key].tobytes() for key in first.fields)
    assert first.soil_no_source_land.tobytes() == second.soil_no_source_land.tobytes()
    assert first.specific_humidity_authority == second.specific_humidity_authority


def test_posted_plan_validates_times_and_complete_population_before_native_work(tmp_path):
    plan = recipe()
    with pytest.raises(ValueError, match="increasing"):
        make_stream(tmp_path/"naive", plan.base, ())
    with pytest.raises(ValueError, match="increasing"):
        make_stream(tmp_path/"repeated", plan.base, (0, 0))
    sources = streams(tmp_path/"sources", plan)
    with pytest.raises(ValueError, match="complete recipe"):
        PostedPhysicalProvider(plan, {plan.base.identity: sources[plan.base.identity]})
    with pytest.raises(ValueError, match="bracket"):
        sources[plan.base.identity].bracket_times(START-timedelta(hours=1))
    assert sources[plan.donor_population[0].identity].bracket_times(START+timedelta(hours=1)) == (
        START, START+timedelta(hours=3))


def test_explicit_control_members_repeat_only_the_unchanged_base(tmp_path):
    plan = recipe()
    control = replace(plan, kind="control", donor_population=(),
                      members=(RecipeMember(4, 4004, plan.base), RecipeMember(9, 4009, plan.base)))
    stream = make_stream(tmp_path/"base", plan.base, (0, 1, 2))
    provider = PostedPhysicalProvider(control, {plan.base.identity: stream})
    assert [member["index"] for member in provider.plan["recipe"]["members"]] == [4, 9]
    other = plan.donor_population[0]
    changed = replace(control, members=(control.members[0], RecipeMember(9, 4009, other)))
    donor_stream = make_stream(tmp_path/"other", other, (0, 3))
    with pytest.raises(ValueError, match="unchanged base"):
        PostedPhysicalProvider(changed, {plan.base.identity: stream, other.identity: donor_stream})


def test_initial_and_later_posted_frames_equal_sealed_native_preparation(tmp_path, bridge):
    from gpuwm.ensemble.physical_recenter import RecenteredPhysicalPreparation
    plan = recipe()
    sources = streams(tmp_path/"sources", plan)
    base = sources[plan.base.identity]
    publish(base, 0)
    offsets = (-2., .25, 1.5)
    for trajectory, offset in zip(plan.donor_population, offsets):
        publish(sources[trajectory.identity], 0, offset)
    provider = PostedPhysicalProvider(plan, sources, amplitude=.7, cpu_bridge=bridge, workers=2)
    outputs0 = {member.index: tmp_path/f"first-{member.index}" for member in plan.members}
    receipts0 = provider.prepare(START, outputs0, work_root=tmp_path/"work0")
    # t0 has completed while every source still lacks its future frame/seal.
    assert all(not (stream.root/"physical-seal.json").exists() for stream in sources.values())
    assert all(not stream._marker(1).exists() for stream in sources.values())
    assert set(receipts0) == {7, 12, 29}
    assert receipts0[12]["receipt"]["member_seed"] == 1012

    waits = []
    def wait(trajectory, valid_time):
        waits.append((trajectory.identity, valid_time))
        hour = int((valid_time-START).total_seconds()/3600)
        offset = 0.0 if trajectory == plan.base else offsets[plan.donor_population.index(trajectory)]*4
        publish(sources[trajectory.identity], hour, offset)
    for stream in sources.values():
        stream.wait = wait
    outputs1 = {member.index: tmp_path/f"later-{member.index}" for member in plan.members}
    provider.prepare(START+timedelta(hours=1), outputs1, work_root=tmp_path/"work1")
    assert len(waits) == 4  # One base knot and the right bracket of all three donors.

    base_path = sealed_store(tmp_path/"sealed-base", base, [snapshot(0), snapshot(1)])
    donors = {source.identity: sealed_store(tmp_path/f"sealed-{source.identity}", sources[source.identity],
                                           [snapshot(0, offset), snapshot(3, offset*4)])
              for source, offset in zip(plan.donor_population, offsets)}
    full = RecenteredPhysicalPreparation(base_path, donors, amplitude=.7, cpu_bridge=bridge, workers=1)
    reference = {member.trajectory.identity: tmp_path/f"reference-{member.index}" for member in plan.members}
    full.prepare(reference)
    for member in plan.members:
        expected = NativePhysicalStore(reference[member.trajectory.identity])
        assert_same_frame(NativePhysicalStore(outputs0[member.index]).read(0), expected.read(0))
        assert_same_frame(NativePhysicalStore(outputs1[member.index]).read(0), expected.read(1))

    # A selected original index uses the same complete population and bytes.
    selected = plan.select_members((29,))
    one = PostedPhysicalProvider(selected, sources, amplitude=.7, cpu_bridge=bridge)
    one.prepare(START+timedelta(hours=1), {29: tmp_path/"single"}, work_root=tmp_path/"single-work")
    assert_same_frame(NativePhysicalStore(tmp_path/"single").read(0),
                      NativePhysicalStore(outputs1[29]).read(0))


def test_missing_unselected_donor_cannot_change_mean_or_publish_outputs(tmp_path, bridge):
    plan = recipe()
    sources = streams(tmp_path/"sources", plan)
    publish(sources[plan.base.identity], 0)
    for trajectory in plan.donor_population[:2]:
        publish(sources[trajectory.identity], 0)
    provider = PostedPhysicalProvider(plan.select_members((7,)), sources, cpu_bridge=bridge)
    with pytest.raises(PhysicalFramePending) as failure:
        provider.prepare(START, {7: tmp_path/"member"}, work_root=tmp_path/"work")
    assert failure.value.trajectory == plan.donor_population[2]
    assert not (tmp_path/"member").exists()
    assert not (tmp_path/"work").exists()


def test_source_failure_propagates_without_member_renumbering(tmp_path, bridge):
    from gpuwm.ingest.boundary_stream import SourceBehind
    plan = recipe()
    sources = streams(tmp_path/"sources", plan)
    publish(sources[plan.base.identity], 0)
    def late(trajectory, valid_time):
        raise SourceBehind({"source": trajectory.source, "lead": 0}, message="planned source is late")
    for trajectory in plan.donor_population:
        sources[trajectory.identity].wait = late
    provider = PostedPhysicalProvider(plan, sources, cpu_bridge=bridge)
    with pytest.raises(SourceBehind, match="planned source"):
        provider.prepare(START, {29: tmp_path/"member"}, work_root=tmp_path/"work")
    assert not (tmp_path/"member").exists()


@pytest.mark.parametrize("mutation", ["head", "marker", "manifest", "payload"])
def test_consumed_physical_authorities_and_arrays_cannot_change(tmp_path, bridge, mutation):
    stream = make_stream(tmp_path/"source", recipe().base, (0, 1))
    publish(stream, 0)
    store, _ = stream.require(START)
    if mutation == "head":
        document = deepcopy(stream.head)
        document["input_plan_sha256"] = "f"*64
        stream.head_path.write_text(json.dumps(document))
    elif mutation == "marker":
        path = stream._marker(0)
        document = json.loads(path.read_text())
        document["decoded_leads"]["0"] = "f"*64
        path.write_text(json.dumps(document))
    elif mutation == "manifest":
        store.manifest_path.write_text(store.manifest_path.read_text()+" ")
    else:
        with (store.root/store.document["frames"][0]["file"]).open("ab") as output:
            output.write(b"changed")
    with pytest.raises(ValueError, match="changed|differs|differ"):
        stream.require(START)


def test_ready_publication_and_complete_seal_are_immutable(tmp_path, bridge):
    stream = make_stream(tmp_path/"source", recipe().base, (0, 1))
    publish(stream, 0)
    with pytest.raises(FileExistsError, match="immutable"):
        publish(stream, 0)
    with pytest.raises(PhysicalFramePending):
        stream.seal()
    assert not (stream.root/"physical-seal.json").exists()
    publish(stream, 1)
    seal = stream.seal()
    assert len(seal["frames"]) == 2
    assert PostedPhysicalStream(stream.root).require(START)[0].read(0).valid_time == snapshot().valid_time


def test_direct_member_frames_preserve_payload_and_original_identity(tmp_path, bridge):
    plan = recipe()
    plan = replace(plan, kind="input-ensemble", donor_population=(), members=plan.members[1:2])
    stream = make_stream(tmp_path/"source", plan.members[0].trajectory, (0, 1, 2))
    publish(stream, 0, 3.)
    provider = PostedPhysicalProvider(plan, {stream.trajectory.identity: stream})
    output = provider.prepare(START, {12: tmp_path/"member"}, work_root=tmp_path/"work")
    assert output[12]["receipt"]["member_index"] == 12
    assert output[12]["receipt"]["member_seed"] == 1012
    assert_same_frame(NativePhysicalStore(tmp_path/"member").read(0), snapshot(0, 3.))


def test_native_unit_authority_still_refuses_bad_donor_before_outputs(tmp_path, bridge):
    plan = recipe()
    sources = streams(tmp_path/"sources", plan)
    bad = plan.donor_population[0]
    contract = analytic_field_contract(GRID)
    contract["arrays"]["field__TT"]["units"] = "degC"
    sources[bad.identity] = make_stream(tmp_path/"wrong-units", bad, (0, 3), contract=contract)
    for stream in sources.values():
        publish(stream, 0)
    provider = PostedPhysicalProvider(plan, sources, cpu_bridge=bridge)
    with pytest.raises(ValueError, match="units K"):
        provider.prepare(START, {7: tmp_path/"member"}, work_root=tmp_path/"work")
    assert not (tmp_path/"member").exists()


def test_posted_source_identity_preserves_explicit_plan_and_native_authorities():
    from gpuwm.ensemble.posted_physical import posted_source_identity, PLAN_REFERENCE_SCHEMA, SOURCE_AUTHORITY_KEY
    from gpuwm.ingest.boundary_stream import as_posted_placeholder, input_plan_sha256
    plan = {"schema": "gpuwm.input-plan.v1", "route_table_sha256": "a"*64,
            "manifest": {"schema": "test", "files": {"lead-0": {"name": "field.grib", "sha256": None}}}}
    placeholder = as_posted_placeholder(input_plan_sha256(plan))
    original = {"input_manifest_sha256": placeholder, "composition_receipt_sha256": placeholder,
                "mapping_sha256": "e"*64, "water_temperature_overlay": {"source": "declared"}}
    value = posted_source_identity(original, input_plan=plan)
    assert value["input_manifest_sha256"]["schema"] == PLAN_REFERENCE_SCHEMA
    assert value["composition_receipt_sha256"]["identity_path"] == ["composition_receipt_sha256"]
    assert value[SOURCE_AUTHORITY_KEY]["input_plan"] == plan
    assert value["mapping_sha256"] == original["mapping_sha256"]
    assert value["water_temperature_overlay"] == original["water_temperature_overlay"]
    assert "as-posted:" not in json.dumps(value)
    reordered = json.loads(json.dumps(original, sort_keys=True))
    assert posted_source_identity(reordered, input_plan=plan) == value
    assert value[SOURCE_AUTHORITY_KEY]["deferred_identity_paths"] == [
        ["composition_receipt_sha256"], ["input_manifest_sha256"]]
    with pytest.raises(ValueError, match="another posted input plan"):
        posted_source_identity({**original, "composition_receipt_sha256": "as-posted:"+"f"*64}, input_plan=plan)


@pytest.fixture
def portable_provider(tmp_path, bridge):
    from types import SimpleNamespace
    from gpuwm.native_wrf_contract import native_geometry_contract
    from gpuwm.static.lambert import LambertGrid
    from gpuwm.ensemble.posted_physical import posted_source_identity
    from gpuwm.ingest.boundary_stream import read_head, posted_lead_marker_sha256
    from test_posted_preparation import _as_posted_tree, _posted_marker
    source_path = tmp_path/"ordinary"
    source_path.mkdir()
    _writer, ordinary_root, _digest = _as_posted_tree(source_path)
    head = read_head(ordinary_root)
    source_plan = head["basis"]["as_posted"]["input_plan"]
    native_source = head["basis"]["cache"]["identity"]["source_identity"]
    cfg = SimpleNamespace(nx=3, ny=2, nz=5, dx=3000., dy=3000.)
    grid = LambertGrid(ref_lat=40, ref_lon=-100, truelat1=30, truelat2=60,
                       stand_lon=-100, dx=3000, dy=3000, e_we=4, e_sn=3)
    geometry = native_geometry_contract(grid, cfg)
    static = physical_static_identity({"HGT_M": np.zeros((2, 3))})
    source = posted_source_identity({**native_source, "static_identity": static}, input_plan=source_plan)
    trajectory = SourceTrajectory("gfs", START)
    plan = SourceRecipe("control", trajectory, START, START+timedelta(hours=2),
                        (RecipeMember(17, 7017, trajectory),))
    stream = PostedPhysicalStream.create(tmp_path/"physical-source", trajectory=trajectory,
        valid_times=[START+timedelta(hours=value) for value in range(3)],
        grid_identity=geometry, source_identity=source,
        field_contract=analytic_field_contract(geometry),
        input_plan_sha256=head["basis"]["as_posted"]["input_plan_sha256"])
    for hour in range(3):
        stream.publish(snapshot(hour), posted_leads={str(hour): posted_lead_marker_sha256(_posted_marker(hour))},
                       decoded_leads={str(hour): "d"*64})
    provider = PostedPhysicalProvider(plan, {trajectory.identity: stream}, cpu_bridge=bridge)
    provider.write_plan(tmp_path/"provider", prepared_roots={trajectory.identity: ordinary_root})
    return provider, grid, cfg, source_plan, native_source, static


def test_serialized_provider_native_binding_and_portable_source_seal(portable_provider, bridge):
    from gpuwm.ensemble.posted_physical import (
        bind_posted_physical_input, validate_posted_physical_input, validate_provider_seal,
        validate_posted_native_source,
    )
    provider, grid, cfg, source_plan, native_source, static = portable_provider
    reopened = PostedPhysicalProvider.open(provider.root, cpu_bridge=bridge)
    bindings = []
    for hour in range(3):
        instant = START+timedelta(hours=hour)
        store, receipt = reopened.resolve(17, instant)
        binding = bind_posted_physical_input(store, receipt, provider_plan=reopened.plan,
            member_index=17, valid_time=instant, grid=grid, cfg=cfg,
            source_identity=native_source, input_plan=source_plan, static_identity=static)
        assert validate_posted_physical_input(binding, provider_plan=provider.plan) == binding
        validate_posted_native_source(binding, source_identity=native_source,
            input_manifest_authority=native_source["input_manifest_sha256"],
            input_plan=source_plan, static_identity=static)
        assert reopened.resolve(17, instant)[1] == receipt
        assert_same_frame(store.read(0), snapshot(hour))
        bindings.append(binding)
    seal = reopened.seal()
    assert validate_provider_seal(seal, provider_plan=reopened.plan,
                                  receipts=[item["provider_receipt"] for item in bindings]) == seal
    assert "as-posted:" not in json.dumps(bindings)
    capsule = next(iter(seal["sources"].values()))
    final_digest = capsule["ordinary_seal"]["as_posted"]["input_manifest_sha256"]
    sealed_identity = {**native_source, "input_manifest_sha256": final_digest}
    validate_posted_native_source(bindings[0], source_identity=sealed_identity,
        input_manifest_authority=final_digest, static_identity=static)
    with pytest.raises(ValueError, match="scientific source authority"):
        validate_posted_native_source(bindings[0], source_identity={**sealed_identity, "adapter": "changed"},
            input_manifest_authority=final_digest, static_identity=static)
    with pytest.raises(ValueError, match="different input plans"):
        validate_posted_native_source(bindings[0], source_identity=native_source,
            input_manifest_authority=native_source["input_manifest_sha256"],
            input_plan={**source_plan, "route_table_sha256": "f"*64})
    changed = deepcopy(bindings[0])
    changed["provider_receipt"]["member_seed"] += 1
    with pytest.raises(ValueError, match="seed"):
        validate_posted_physical_input(changed, provider_plan=provider.plan)


@pytest.mark.parametrize("mutation", ["raw_manifest", "physical_marker", "artifact_path"])
def test_portable_source_capsule_cannot_bypass_ordinary_final_authorities(portable_provider, mutation):
    from gpuwm.ensemble.posted_physical import validate_provider_seal
    from gpuwm.ingest.boundary_stream import BoundaryStreamError
    provider, *_ = portable_provider
    seal = deepcopy(provider.seal())
    source = next(iter(seal["sources"].values()))
    if mutation == "raw_manifest":
        name = "source-input-manifest.json"
        manifest = json.loads(source["artifacts"][name])
        manifest["files"]["grib-f001"]["sha256"] = "f"*64
        source["artifacts"][name] = json.dumps(manifest)
    elif mutation == "physical_marker":
        marker = source["physical_frames"][1]["marker"]
        marker["posted_leads"]["1"] = "f"*64
        import hashlib
        source["physical_frames"][1]["marker_sha256"] = hashlib.sha256(
            (json.dumps(marker, sort_keys=True, separators=(",", ":"))+"\n").encode()).hexdigest()
    else:
        source["artifacts"]["../outside.json"] = "{}"
    with pytest.raises((ValueError, BoundaryStreamError)):
        validate_provider_seal(seal, provider_plan=provider.plan)


@pytest.mark.parametrize("mutation", ["member_index", "source_head", "recipe_digest", "amplitude"])
def test_provider_plan_checks_its_own_roster_and_source_hashes(portable_provider, mutation):
    import hashlib
    from gpuwm.ensemble.posted_physical import validate_provider_plan
    provider, *_ = portable_provider
    plan = deepcopy(provider.plan)
    if mutation == "member_index":
        plan["recipe"]["members"].append(deepcopy(plan["recipe"]["members"][0]))
        plan["recipe_sha256"] = hashlib.sha256(json.dumps(
            plan["recipe"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    elif mutation == "source_head":
        next(iter(plan["streams"].values()))["head_sha256"] = "f"*64
    elif mutation == "recipe_digest":
        plan["recipe_sha256"] = "f"*64
    else:
        plan["amplitude"] = -1.
    with pytest.raises(ValueError):
        validate_provider_plan(plan)


def test_provider_receipt_cannot_relabel_a_source_frame_index(portable_provider):
    import hashlib
    from gpuwm.ensemble.posted_physical import validate_provider_receipt
    provider, *_ = portable_provider
    _store, receipt = provider.resolve(17, START)
    receipt = deepcopy(receipt)
    frame = next(iter(receipt["sources"].values()))[0]
    frame["marker"]["index"] = 19
    frame["marker_sha256"] = hashlib.sha256((json.dumps(
        frame["marker"], sort_keys=True, separators=(",", ":"))+"\n").encode()).hexdigest()
    with pytest.raises(ValueError, match="consumed source marker"):
        validate_provider_receipt(receipt, provider_plan=provider.plan)


@pytest.mark.parametrize("inner_authority", ["absent", "conflict"])
def test_source_capsule_accepts_outer_manifest_binding_and_preserves_inner_conflicts(portable_provider, inner_authority):
    from gpuwm.ensemble.posted_physical import validate_provider_seal
    from gpuwm.ingest.boundary_stream import head_sha256, proof_document_name
    provider, *_ = portable_provider
    seal = deepcopy(provider.seal())
    certificate = next(iter(seal["sources"].values()))
    artifacts = certificate["artifacts"]
    head = json.loads(artifacts["boundary-stream/head.json"])
    header_name = head["basis"]["cache"]["directory"]+"/header.json"
    header = json.loads(artifacts[header_name])
    # Both are valid ordinary identities: some native adapters carry the raw
    # manifest only in the outer source_manifest_sha256. The conflicting
    # inner value is intentionally fixed across head and seal, so the shared
    # physical-source correspondence must reject it explicitly.
    for identity in (head["basis"]["cache"]["identity"], header["identity"]):
        identity["source_identity"].pop("input_manifest_sha256")
        if inner_authority == "conflict":
            identity["source_identity"]["input_manifest_sha256"] = "f"*64
    head["head_sha256"] = head_sha256(head)
    basis = {key: header[key] for key in ("schema", "identity", "metadata", "arrays", "payload_bytes")}
    import hashlib
    header["content_sha256"] = hashlib.sha256(json.dumps(basis, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    artifacts["boundary-stream/head.json"] = json.dumps(head)
    artifacts[header_name] = json.dumps(header)
    for name in tuple(artifacts):
        if name == "boundary-stream/posted-leads.json" or name.startswith("boundary-stream/segments/"):
            document = json.loads(artifacts[name])
            document["head_sha256"] = head["head_sha256"]
            artifacts[name] = json.dumps(document)
    proof_name = proof_document_name(head)
    proof = json.loads(artifacts[proof_name])
    proof["boundary_stream"]["head_sha256"] = head["head_sha256"]
    proof["prepared_cache"]["content_sha256"] = header["content_sha256"]
    artifacts[proof_name] = json.dumps(proof)
    certificate["ordinary_seal"].update(
        head_sha256=head["head_sha256"], content_sha256=header["content_sha256"],
        proof_sha256=hashlib.sha256(artifacts[proof_name].encode()).hexdigest())
    certificate["ordinary_seal"]["as_posted"]["identity_changed"] = ["source_manifest_sha256"]
    if inner_authority == "absent":
        assert validate_provider_seal(seal, provider_plan=provider.plan) == seal
    else:
        with pytest.raises(ValueError, match="deferred input manifest"):
            validate_provider_seal(seal, provider_plan=provider.plan)


def test_source_context_reuses_pinned_ordinary_head_and_waits(portable_provider, bridge):
    provider, *_ = portable_provider
    reopened = PostedPhysicalProvider.open(provider.root, cpu_bridge=bridge)
    context = reopened.source_context(17)
    identity = provider.recipe.base.identity
    assert context.trajectory == provider.recipe.base
    assert context.prepared_root == provider.prepared_roots[identity]
    assert reopened.prepared_headpins == provider.prepared_headpins
    assert context.prepared_head["head_sha256"] == provider.prepared_headpins[identity]
    assert context.source_plan == context.prepared_head["basis"]["as_posted"]["input_plan"]
    assert context.physical_stream.head_sha256 == provider.streams[identity].head_sha256
    assert context.require_interval(0)["index"] == 0
    context.wait_sealed()
    capsule = context.capture_seal()
    assert capsule["ordinary_seal"]["head_sha256"] == provider.prepared_headpins[identity]
    with pytest.raises(ValueError, match="original RecipeMember"):
        reopened.source_context(0)


@pytest.mark.parametrize("reopen", [False, True])
@pytest.mark.parametrize("wait_kind", ["context", "physical", "seal"])
def test_late_member_wait_observer_stops_only_that_member(
        portable_provider, tmp_path, bridge, reopen, wait_kind):
    from gpuwm.ensemble.posted_native import writer_source_wait
    from gpuwm.ingest.boundary_stream import (
        BoundaryStreamStopped, PreparedTreeWriter, STOP_NAME, segment_marker_path,
    )
    provider, *_ = portable_provider
    if reopen:
        provider = PostedPhysicalProvider.open(provider.root, cpu_bridge=bridge)
    context = provider.source_context(17)
    # Retain a reader created before the writer exists. Its observer must
    # still see the member's later cancellation, as must physical-frame waits.
    intervals = context.intervals()
    writer = PreparedTreeWriter(staging=tmp_path/"member-staging", output_root=tmp_path/"member",
        identity=context.prepared_head["basis"]["cache"]["identity"])
    writer.stream_path.mkdir(parents=True, exist_ok=True)
    (writer.stream_path/STOP_NAME).write_text(json.dumps({"reason": "member cancelled"}))
    provider.set_wait_observer(writer_source_wait(writer))
    source_stop = context.prepared_root/"boundary-stream"/STOP_NAME
    assert not source_stop.exists()
    if wait_kind == "seal":
        (context.prepared_root/"proof.json").unlink()
        action = context.wait_sealed
    else:
        segment_marker_path(context.prepared_root, 0).unlink()
        action = (lambda: intervals.require(0)) if wait_kind == "context" else (
            lambda: context.physical_stream.wait(context.trajectory, context.physical_stream.times[1]))
    with pytest.raises(BoundaryStreamStopped, match="member cancelled"):
        action()
    assert not source_stop.exists()


def test_recentered_source_context_selects_base_not_donor(tmp_path, monkeypatch):
    plan = recipe()
    provider = PostedPhysicalProvider(plan, streams(tmp_path/"sources", plan))
    monkeypatch.setattr(provider, "_source_context", lambda identity: identity)
    assert provider.source_context(29) == plan.base.identity
    direct = replace(plan, kind="input-ensemble", donor_population=())
    donor_streams = {key: value for key, value in provider.streams.items() if key != plan.base.identity}
    selected = PostedPhysicalProvider(direct, donor_streams)
    monkeypatch.setattr(selected, "_source_context", lambda identity: identity)
    assert selected.source_context(29) == plan.members[2].trajectory.identity


def test_source_context_refuses_replaced_prepared_head_with_same_input_plan(portable_provider):
    from gpuwm.ingest.boundary_stream import BoundaryStreamError, head_sha256
    provider, *_ = portable_provider
    context = provider.source_context(17)
    path = context.prepared_root/"boundary-stream/head.json"
    document = json.loads(path.read_text())
    document["basis"]["proof_head"]["unrelated_change"] = True
    document["head_sha256"] = head_sha256(document)
    path.write_text(json.dumps(document))
    with pytest.raises(BoundaryStreamError, match="pinned head"):
        context.require_interval(0)
    with pytest.raises(BoundaryStreamError, match="pinned head"):
        provider.source_context(17)
    with pytest.raises(BoundaryStreamError, match="pinned head"):
        provider.seal()
