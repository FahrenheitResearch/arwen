"""Native HRRR trees bind the root's live leads before its documents seal."""

import json

import pytest

from gpuwm import hrrr_hierarchy_direct as hierarchy
from gpuwm.ingest import boundary_stream
from test_native_hrrr_posted import (
    BRIDGE, DOCUMENT_KEYS, LEADS, SOURCE, _decoded, _native_posted,
)


def _root(tmp_path):
    root, digests = _native_posted(tmp_path)
    head = boundary_stream.read_head(root)
    return root, head, digests


def _binding(root, head, requested=None, identity=None, source=None):
    return hierarchy._root_source_binding(
        paths=hierarchy._root_paths(root),
        source_manifest=root / SOURCE if source is None else source,
        source_manifest_sha256=(head["basis"]["cache"]["identity"][
            "source_manifest_sha256"] if requested is None else requested),
        identity=head["basis"]["cache"]["identity"]
        if identity is None else identity,
        root_head=head, root_preparation=root)


def test_live_tree_binds_the_plan_before_either_document_exists(tmp_path):
    root, head, _ = _root(tmp_path)
    (root / BRIDGE).unlink()
    (root / SOURCE).unlink()
    placeholder = boundary_stream.as_posted_placeholder(
        head["basis"]["as_posted"]["input_plan_sha256"])
    assert _binding(root, head) == (placeholder, placeholder)


@pytest.mark.parametrize("key", DOCUMENT_KEYS)
def test_live_tree_refuses_a_document_identity_outside_its_plan(tmp_path, key):
    root, head, _ = _root(tmp_path)
    identity = {**head["basis"]["cache"]["identity"], key: "a" * 64}
    with pytest.raises(ValueError, match="differs from its input plan"):
        _binding(root, head, identity=identity)


def test_live_tree_refuses_an_unrelated_source_manifest_path(tmp_path):
    root, head, _ = _root(tmp_path)
    with pytest.raises(ValueError, match="source manifest differs"):
        _binding(root, head, source=tmp_path / "elsewhere" / "SHA256SUMS")


def test_tree_head_keeps_the_exact_root_plan_and_start_marker_bindings(tmp_path):
    _, head, _ = _root(tmp_path)
    adapted = hierarchy._posted_tree_head_inputs(head)
    root_posted = head["basis"]["as_posted"]
    assert adapted["input_plan"] == root_posted["input_plan"]
    assert adapted["start_marker_sha256"] == root_posted["start_marker_sha256"]
    assert adapted["documents"] == root_posted["documents"]
    assert adapted["document_bound_identity_keys"] == DOCUMENT_KEYS
    assert adapted["forcing_leads"] == list(LEADS)
    assert set(adapted["seal_authored_proof_keys"]) == {
        "provenance", "input_manifest_sha256", "posting"}
    adapted["input_plan"]["manifest"]["source"]["forecast_hours"].append(99)
    assert 99 not in root_posted["input_plan"]["manifest"]["source"][
        "forecast_hours"]


def test_tree_copies_the_documents_verified_by_the_root_seal(tmp_path):
    root, head, digests = _root(tmp_path)
    sealed = boundary_stream.verify_seal(root, head=head)
    target = tmp_path / "tree"
    manifest, documents = hierarchy._copy_posted_root_inputs(
        root, target, head, sealed)
    assert manifest == digests["manifest"]
    assert documents == {"bridge_manifest_sha256": digests["bridge"],
                         "source_manifest_sha256": digests["source"]}
    for name in ("source-input-manifest.json", BRIDGE, SOURCE):
        assert (target / name).read_bytes() == (root / name).read_bytes()


@pytest.mark.parametrize("name", ("source-input-manifest.json", BRIDGE, SOURCE))
def test_tree_refuses_an_input_changed_after_the_root_seal(tmp_path, name):
    root, head, _ = _root(tmp_path)
    sealed = boundary_stream.verify_seal(root, head=head)
    path = root / name
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed after its root seal"):
        hierarchy._copy_posted_root_inputs(root, tmp_path / "tree", head, sealed)


def test_a_root_that_sealed_during_hierarchy_launch_resolves_its_placeholder(
        tmp_path):
    root, head, digests = _root(tmp_path)
    placeholder = head["basis"]["cache"]["identity"]["source_manifest_sha256"]
    assert hierarchy._settled_source_manifest_sha(
        root, root / SOURCE, placeholder) == digests["source"]
    with pytest.raises(ValueError, match="differs from its input plan"):
        hierarchy._settled_source_manifest_sha(
            root, root / SOURCE,
            boundary_stream.as_posted_placeholder("a" * 64))


def test_a_sealed_manifest_digest_keeps_its_original_binding(tmp_path):
    assert hierarchy._settled_source_manifest_sha(
        tmp_path / "absent", tmp_path / "absent" / "SHA256SUMS", "a" * 64
    ) == "a" * 64


@pytest.mark.parametrize("foreign", (False, True))
def test_tree_reads_the_owned_decoder_staging_before_the_bridge_is_published(
        tmp_path, foreign):
    root = tmp_path / "root"
    bridge = root / "native/native-bridge"
    staging = (tmp_path / ".native-bridge.partial-123" if foreign else
               bridge.with_name(".native-bridge.partial-123"))
    staging.mkdir(parents=True)
    signal = root / "native/pipeline-signals/preflight.ready"
    signal.parent.mkdir(parents=True)
    signal.write_text(
        f"status\tPASS\nstaging_retention\tuntil_consumer_finish\n"
        f"canonical_output\t{bridge}\nstaging_root\t{staging}\n")
    if foreign:
        with pytest.raises(ValueError, match="staging differs"):
            hierarchy._posted_start_bridge(root, bridge)
    else:
        assert hierarchy._posted_start_bridge(root, bridge) == staging


def test_tree_heartbeat_relays_source_waits_and_clears_the_arrived_lead():
    from types import SimpleNamespace

    calls = []
    writer = SimpleNamespace(
        waiting_for_source=lambda cause: calls.append(("source", cause)),
        source_arrived=lambda: calls.append(("producing", None)))
    cause = {"on": "source", "source": "hrrr", "lead": 8,
             "expected_at": "2026-09-30T13:00:00Z"}
    hierarchy._relay_root_wait(writer, {"cause": cause})
    hierarchy._relay_root_wait(writer, {"cause": None})
    hierarchy._relay_root_wait(writer, None)
    assert calls == [("source", cause), ("producing", None),
                     ("producing", None)]


@pytest.mark.parametrize("changed", (False, True))
def test_tree_maps_only_the_decoded_start_bytes_bound_by_the_root(
        tmp_path, monkeypatch, changed):
    from gpuwm.ingest import hrrr
    from tools import hrrr_single_domain_benchmark as benchmark

    root, head, _ = _root(tmp_path)
    expected = _decoded(LEADS[0])
    record = {**expected, **({"./extra": "a" * 64} if changed else {})}
    monkeypatch.setattr(benchmark, "_decoded_lead_record",
                        lambda *_args, **_kwargs: record)
    calls = []
    monkeypatch.setattr(hrrr, "load_hrrr_pipeline_ready_window",
                        lambda bridge, lead: calls.append((bridge, lead)) or "snapshot")
    if changed:
        with pytest.raises(ValueError, match="differs from its root interval"):
            hierarchy._posted_start_snapshots(root, root / "native/native-bridge",
                                               head, LEADS[0])
        assert not calls
    else:
        assert hierarchy._posted_start_snapshots(
            root, root / "native/native-bridge", head, LEADS[0]) == ("snapshot",)
        assert calls == [(root / "native/native-bridge", LEADS[0])]


def test_native_tree_relay_seals_the_same_boundary_cache_under_document_digests(
        tmp_path):
    from test_boundary_stream import _initial, _met

    root, root_head, digests = _root(tmp_path)
    root_seal = boundary_stream.verify_seal(root, head=root_head)
    stream = boundary_stream.StreamedIntervals(root, head=root_head)
    destination = tmp_path / "tree"
    staging = tmp_path / ".tree-staging"
    writer = boundary_stream.PreparedTreeWriter(
        staging=staging, output_root=destination,
        identity=root_head["basis"]["cache"]["identity"],
        cache_name="hierarchy-head/domains/d01/prepared-cache",
        proof_name="receipt.json")
    writer.write_head(
        initial_result=_initial(), met=_met(),
        lbc=root_head["basis"]["cache"]["lbc"],
        proof_head={"schema": "tree"},
        as_posted=hierarchy._posted_tree_head_inputs(root_head))
    for index in range(len(stream)):
        writer.write_segment(index, stream[index],
                             relay_marker=stream.consumed_markers()[index])
        stream.release(index)
    manifest, documents = hierarchy._copy_posted_root_inputs(
        root, writer.root, root_head, root_seal)
    leads = json.loads((boundary_stream.stream_dir(root)
                        / boundary_stream.POSTED_LEADS_NAME).read_text())
    writer.write_posted_leads(
        {int(lead): row["marker"] for lead, row in leads["leads"].items()},
        route_table_sha256=leads["route_table_sha256"],
        decoded={int(lead): row["decoded"]
                 for lead, row in leads["leads"].items()})
    identity = {**root_head["basis"]["cache"]["identity"], **documents}
    cache = writer.seal_cache(identity=identity, manifest_sha256=manifest,
                              document_sha256=documents)
    writer.publish({"schema": "tree", "input_manifest_sha256": manifest,
                    "provenance": documents,
                    "posting": {"as_posted": True},
                    "prepared_cache": {"content_sha256": cache["content_sha256"]},
                    "boundary_stream": writer.boundary_stream_proof()})
    head = boundary_stream.read_head(destination)
    sealed = boundary_stream.verify_seal(destination, head=head)
    assert sealed["as_posted"]["input_manifest_sha256"] == digests["manifest"]
    assert sealed["content_sha256"] == root_seal["content_sha256"]
    assert head["basis"]["as_posted"]["start_marker_sha256"] \
        == root_head["basis"]["as_posted"]["start_marker_sha256"]
