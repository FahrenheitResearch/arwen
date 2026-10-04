"""Member input authority remains bound from the first knot through the seal."""
from copy import deepcopy
from datetime import timedelta
import json

import pytest

from gpuwm.ensemble import physical_boundary as authority
from gpuwm.ensemble.posted_physical import bind_posted_physical_input
from gpuwm.ingest.boundary_stream import (
    BoundaryStreamError, PreparedTreeWriter, StreamedIntervals, read_head,
    segment_marker_path, verify_seal,
)
from gpuwm.ingest.lateral_bc import StateBoundaryFrames
from test_boundary_stream import _initial, _met, _snapshots
from test_ensemble_posted_physical import START, bridge, portable_provider


@pytest.fixture
def physical_inputs(portable_provider):
    provider, grid, cfg, source_plan, native_source, static = portable_provider
    bindings = {}
    for index in range(3):
        instant = START + timedelta(hours=index)
        store, receipt = provider.resolve(17, instant)
        bindings[index] = bind_posted_physical_input(
            store, receipt, provider_plan=provider.plan, member_index=17,
            valid_time=instant, grid=grid, cfg=cfg, source_identity=native_source,
            input_plan=source_plan, static_identity=static)
    return provider, bindings, native_source


def _member_tree(tmp_path, physical_inputs, *, finish=True, relay=False):
    provider, bindings, native_source = physical_inputs
    staging = tmp_path / "member-staging"
    staging.mkdir()
    root = tmp_path / "member-prepared"
    writer = PreparedTreeWriter(staging=staging, output_root=root, chained=True,
        identity={"source_identity": {
            **native_source, "ensemble_posted_physical_input": bindings[0]}})
    writer.bind_physical_receipts(bindings)
    snapshots = _snapshots(3)
    times = [START + timedelta(hours=index) for index in range(3)]
    frames = StateBoundaryFrames(spec_bdy_width=5, spec_zone=1, relax_zone=4)
    frames.add_snapshot(snapshots[0], index=0)
    writer.write_head(initial_result=_initial(), met=_met(),
        lbc={"spec_bdy_width": 5, "spec_zone": 1, "relax_zone": 4,
             "schedule": [[0, 3600], [3600, 7200]], "fields": frames.inventory},
        proof_head={"schema": "member-boundary-test"}, input_manifest_sha256="0" * 64,
        forcing=frames, ensemble_physical={"provider_plan": provider.plan,
            "member_index": 17, "initial_receipt": bindings[0]})
    for index in range(1, 3):
        frames.add_snapshot(snapshots[index], index=index)
        interval = frames.interval(index - 1, times)
        marker = None
        if relay:
            from gpuwm.ingest.boundary_stream import SEGMENT_SCHEMA
            marker = {"schema": SEGMENT_SCHEMA, "index": index - 1,
                      "start_seconds": float(interval.start_seconds),
                      "end_seconds": float(interval.end_seconds),
                      "fields": list(interval.fields)}
        writer.write_segment(index - 1, interval, relay_marker=marker)
        frames.release(index - 1)
    if finish:
        receipt = writer.seal_cache(physical_provider_seal=provider.seal())
        writer.publish({"schema": "member-boundary-test", "prepared_cache": {
            "content_sha256": receipt["content_sha256"]},
            "boundary_stream": writer.boundary_stream_proof()})
    return writer, root


def test_member_boundary_catalog_replays_exact_consumed_knots(tmp_path, physical_inputs):
    _writer, root = _member_tree(tmp_path, physical_inputs)
    head = read_head(root)
    source = StreamedIntervals(root, head=head)
    source.require(0)
    source.require(1)
    verify_seal(root, head=head)
    header = json.loads((root / "prepared-cache" / "header.json").read_text())
    catalog = header["metadata"]["user"][authority.KEY]
    assert set(catalog) == {"schema", "provider_plan_sha256", "member_index",
                            "knot/0", "knot/1", "knot/2", "provider_seal"}
    assert catalog["member_index"] == 17
    assert [catalog[f"knot/{index}"]["binding"] for index in range(3)] == list(physical_inputs[1].values())


def test_member_cannot_finish_without_every_source_seal(tmp_path, physical_inputs):
    writer, _root = _member_tree(tmp_path, physical_inputs, finish=False)
    try:
        with pytest.raises(ValueError, match="every source seal"):
            writer.seal_cache()
        with pytest.raises(ValueError, match="checked native writer"):
            writer.seal_cache(completed_metadata={authority.KEY: {}},
                              physical_provider_seal=physical_inputs[0].seal())
    finally:
        writer.fail(RuntimeError("expected incomplete-source qualification"))


def test_ordinary_source_relay_keeps_member_physical_endpoints(tmp_path, physical_inputs):
    _writer, root = _member_tree(tmp_path, physical_inputs, relay=True)
    head = read_head(root)
    source = StreamedIntervals(root, head=head)
    for index in range(2):
        marker = source.require(index)
        assert marker[authority.KEY][str(index)]["binding"] == physical_inputs[1][index]
        assert marker[authority.KEY][str(index + 1)]["binding"] == physical_inputs[1][index + 1]
    verify_seal(root, head=head)


@pytest.mark.parametrize("mutation", ["missing", "seed", "record_digest", "knot"])
def test_member_segment_refuses_missing_or_changed_input(tmp_path, physical_inputs, mutation):
    _writer, root = _member_tree(tmp_path, physical_inputs)
    marker_path = segment_marker_path(root, 1)
    marker = json.loads(marker_path.read_text())
    records = marker[authority.KEY]
    if mutation == "missing":
        del marker[authority.KEY]
    elif mutation == "seed":
        records["2"]["binding"]["provider_receipt"]["member_seed"] += 1
        records["2"]["sha256"] = authority.digest(records["2"]["binding"])
    elif mutation == "record_digest":
        records["2"]["sha256"] = "f" * 64
    else:
        records["2"] = deepcopy(records["1"])
    marker_path.write_text(json.dumps(marker))
    source = StreamedIntervals(root, head=read_head(root))
    source.require(0)
    with pytest.raises(BoundaryStreamError):
        source.require(1)
    with pytest.raises((BoundaryStreamError, ValueError)):
        verify_seal(root, head=read_head(root))


def test_head_preserves_original_member_and_native_clock(physical_inputs):
    provider, bindings, _source = physical_inputs
    spec = {"provider_plan": provider.plan, "member_index": 17,
            "initial_receipt": bindings[0]}
    head = authority.make_head(spec, [[0, 3600], [3600, 7200]])
    authority.validate_head(head, [[0, 3600], [3600, 7200]])
    with pytest.raises(ValueError):
        authority.make_head({**spec, "member_index": 0}, [[0, 3600], [3600, 7200]])
    with pytest.raises(ValueError, match="contiguous"):
        authority.make_head(spec, [[0, 3600], [3601, 7200]])
    with pytest.raises(ValueError, match="complete recipe window"):
        authority.make_head(spec, [[0, 3600]])


def test_consumed_input_records_do_not_alias_mutable_markers(physical_inputs):
    provider, bindings, _source = physical_inputs
    head = authority.make_head({"provider_plan": provider.plan, "member_index": 17,
        "initial_receipt": bindings[0]}, [[0, 3600], [3600, 7200]])
    writer_seen = {}
    records = authority.segment_records(head, 0, bindings, writer_seen)
    writer_frozen = deepcopy(writer_seen)
    seen = {}
    authority.validate_segment(head, 0, records, seen)
    frozen = deepcopy(seen)
    records["1"]["binding"]["provider_receipt"]["member_seed"] += 1
    assert seen == frozen
    assert writer_seen == writer_frozen


def test_catalog_cannot_replace_a_consumed_input(physical_inputs):
    provider, bindings, _source = physical_inputs
    head = authority.make_head({"provider_plan": provider.plan, "member_index": 17,
        "initial_receipt": bindings[0]}, [[0, 3600], [3600, 7200]])
    seen = {}
    authority.segment_records(head, 0, bindings, seen)
    authority.segment_records(head, 1, bindings, seen)
    catalog = authority.complete_catalog(head, seen, provider.seal())
    catalog["knot/2"] = deepcopy(catalog["knot/1"])
    with pytest.raises(ValueError, match="exact boundary endpoints"):
        authority.validate_catalog(head, catalog, seen)


def test_preflight_metadata_requires_the_exact_verified_physical_catalog(physical_inputs):
    from gpuwm.prepared_single_domain_forecast import _validate_cache_metadata
    from test_prepared_single_domain_forecast import _cache_metadata_case
    provider, bindings, _source = physical_inputs
    head = authority.make_head({"provider_plan": provider.plan, "member_index": 17,
        "initial_receipt": bindings[0]}, [[0, 3600], [3600, 7200]])
    catalog = authority.head_catalog(head)
    case = _cache_metadata_case(user_extra={authority.KEY: catalog})
    with pytest.raises(ValueError, match="user metadata differs"):
        _validate_cache_metadata(**case)
    _validate_cache_metadata(**case, physical_catalog=catalog)
    altered = deepcopy(catalog)
    altered["member_index"] = 18
    with pytest.raises(ValueError, match="user metadata differs"):
        _validate_cache_metadata(**case, physical_catalog=altered)
    case["reader"].header["metadata"]["user"]["unbound_extra"] = True
    with pytest.raises(ValueError, match="user metadata differs"):
        _validate_cache_metadata(**case, physical_catalog=catalog)
