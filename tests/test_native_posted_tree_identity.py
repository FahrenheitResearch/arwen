"""A native tree relays source records and seals its children's identities."""

from __future__ import annotations

import hashlib
import json
import shutil

import pytest

from gpuwm.ingest import boundary_stream


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _rewrite_child(root, head, folder, *, identity=None, user=None):
    directory = root / folder / "domains" / "d02"
    path = directory / "prepared-cache" / "header.json"
    header = json.loads(path.read_text())
    if identity is not None:
        header["identity"] = identity
    if user is not None:
        header["metadata"]["user"] = user
    header["content_sha256"] = boundary_stream._header_content_sha256(header)
    _write_json(path, header)
    receipt_path = directory / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["artifacts"]["prepared_cache"]["content_sha256"] = header[
        "content_sha256"]
    _write_json(receipt_path, receipt)
    if folder == boundary_stream.HIERARCHY_HEAD_DIRNAME:
        head["basis"]["tree"]["children_receipts"]["d02"] = hashlib.sha256(
            receipt_path.read_bytes()).hexdigest()
    return header


def _document_children(tmp_path, *, posted_user=False):
    from test_posted_preparation import _child_tree

    head, _old_manifest, placeholder, identity = _child_tree(tmp_path)
    manifest = {"files": {
        "bridge": {"name": "bridge/SHA256SUMS", "sha256": "a" * 64},
        "source_manifest": {"name": "fetch/SHA256SUMS", "sha256": "b" * 64},
    }}
    manifest_path = tmp_path / "source-input-manifest.json"
    _write_json(manifest_path, manifest)
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    posted = head["basis"]["as_posted"]
    posted.update({
        "manifest_path": manifest_path.name,
        "manifest_bound_identity_keys": ["input_manifest_sha256"],
        "document_bound_identity_keys": {
            "bridge_manifest_sha256": "bridge",
            "source_manifest_sha256": "source_manifest"},
    })
    identity["bridge_manifest_sha256"] = "a" * 64
    identity["source_manifest_sha256"] = "b" * 64
    identity["source_identity"]["input_manifest_sha256"] = digest
    _rewrite_child(tmp_path, head, boundary_stream.SEALED_HIERARCHY_DIRNAME,
                   identity=identity)
    if posted_user:
        posted["posted_user_metadata"] = ["source_manifest_sha256"]
        _rewrite_child(tmp_path, head, boundary_stream.HIERARCHY_HEAD_DIRNAME,
                       user={"grid": 2, "source_manifest_sha256": placeholder})
        _rewrite_child(tmp_path, head, boundary_stream.SEALED_HIERARCHY_DIRNAME,
                       user={"grid": 2, "source_manifest_sha256": "b" * 64})
    return head, digest, identity


@pytest.mark.parametrize("posted_user", [False, True])
def test_native_children_seal_distinct_document_digests(tmp_path, posted_user):
    head, manifest, _identity = _document_children(
        tmp_path, posted_user=posted_user)
    found = boundary_stream.verify_as_posted_tree_children(
        tmp_path, head=head, manifest_sha256=manifest)
    assert found["d02"]["identity_changed"] == [
        "bridge_manifest_sha256", "source_identity.input_manifest_sha256",
        "source_manifest_sha256"]
    assert found["d02"]["head_content_sha256"] \
        != found["d02"]["sealed_content_sha256"]


def test_a_child_cannot_substitute_the_input_manifest_for_a_document(tmp_path):
    head, manifest, identity = _document_children(tmp_path)
    identity["bridge_manifest_sha256"] = manifest
    _rewrite_child(tmp_path, head, boundary_stream.SEALED_HIERARCHY_DIRNAME,
                   identity=identity)
    with pytest.raises(boundary_stream.BoundaryStreamError,
                       match="not the document the sealed input manifest"):
        boundary_stream.verify_as_posted_tree_children(
            tmp_path, head=head, manifest_sha256=manifest)


def test_document_bound_child_check_hashes_the_manifest_it_reads(tmp_path):
    head, manifest, _identity = _document_children(tmp_path)
    with (tmp_path / "source-input-manifest.json").open("a") as stream:
        stream.write("\n")
    with pytest.raises(boundary_stream.BoundaryStreamError,
                       match="not the sealed input manifest"):
        boundary_stream.verify_as_posted_tree_children(
            tmp_path, head=head, manifest_sha256=manifest)


def test_posted_child_metadata_cannot_rewrite_a_start_time_fact(tmp_path):
    head, manifest, _identity = _document_children(tmp_path, posted_user=True)
    _rewrite_child(tmp_path, head, boundary_stream.SEALED_HIERARCHY_DIRNAME,
                   user={"grid": 3, "source_manifest_sha256": "b" * 64})
    with pytest.raises(boundary_stream.BoundaryStreamError,
                       match="sealed d02 differs.*metadata"):
        boundary_stream.verify_as_posted_tree_children(
            tmp_path, head=head, manifest_sha256=manifest)


def _relay_writer(tmp_path, *, start=None, raw_start=False):
    from test_boundary_stream import _initial, _met
    from test_native_hrrr_posted import _native_posted

    source, digests = _native_posted(tmp_path / "source")
    source_head = boundary_stream.read_head(source)
    posted = dict(source_head["basis"]["as_posted"])
    if start is not None:
        posted["start_marker_sha256"] = start
    if raw_start:
        posted["start_markers"] = {}
    staging = tmp_path / ".tree"
    staging.mkdir()
    writer = boundary_stream.PreparedTreeWriter(
        staging=staging, output_root=tmp_path / "tree", chained=False,
        identity=source_head["basis"]["cache"]["identity"])
    writer.write_head(
        initial_result=_initial(), met=_met(),
        lbc=source_head["basis"]["cache"]["lbc"],
        proof_head=source_head["basis"]["proof_head"], as_posted=posted)
    return writer, source, source_head, digests


def test_native_root_intervals_relay_before_the_tree_has_raw_lead_records(
        tmp_path):
    writer, source, source_head, digests = _relay_writer(tmp_path)
    assert writer.head["basis"]["as_posted"]["start_marker_sha256"] \
        == source_head["basis"]["as_posted"]["start_marker_sha256"]
    intervals = boundary_stream.StreamedIntervals(source, head=source_head)
    for k in range(len(intervals)):
        interval = intervals[k]
        consumed = intervals.consumed_markers()[k]
        marker = writer.write_segment(k, interval, relay_marker=consumed)
        assert marker["posted_leads"] == consumed["posted_leads"]
        assert marker["decoded_leads"] == consumed["decoded_leads"]
        intervals.release(k)
    boundary_stream.verify_seal(
        source, head=source_head, consumed=intervals.consumed_markers())
    source_record = json.loads((boundary_stream.stream_dir(source)
                               / boundary_stream.POSTED_LEADS_NAME).read_text())
    writer.write_posted_leads(
        {int(lead): row["marker"]
         for lead, row in source_record["leads"].items()},
        route_table_sha256=source_record["route_table_sha256"],
        decoded={int(lead): row["decoded"]
                 for lead, row in source_record["leads"].items()})
    manifest_path = source_head["basis"]["as_posted"]["manifest_path"]
    for relative in [manifest_path, *[
            row["path"] for row in source_head["basis"]["as_posted"][
                "documents"].values()]]:
        target = writer.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
    source_header = json.loads((source / "native" / "prepared-cache"
                                / "header.json").read_text())
    receipt = writer.seal_cache(
        identity=source_header["identity"], manifest_sha256=digests["manifest"],
        document_sha256={"bridge_manifest_sha256": digests["bridge"],
                         "source_manifest_sha256": digests["source"]})
    source_proof = json.loads((source / "proof.json").read_text())
    writer.publish({**source_proof,
                    "prepared_cache": {"content_sha256": receipt[
                        "content_sha256"]},
                    "boundary_stream": writer.boundary_stream_proof()})
    sealed = boundary_stream.verify_seal(writer.root, head=writer.head)
    assert sealed["content_sha256"] == source_header["content_sha256"]
    for spec in source_header["arrays"].values():
        assert (source / "native" / "prepared-cache" / spec["file"]).read_bytes() \
            == (writer.cache_path / spec["file"]).read_bytes()


@pytest.mark.parametrize("name", ["posted_leads", "decoded_leads"])
@pytest.mark.parametrize("bad", [None, {"6": "a" * 64},
                                {"6": "bad", "7": "b" * 64}])
def test_a_relay_cannot_drop_or_replace_its_two_lead_record_digests(
        tmp_path, name, bad):
    writer, source, source_head, _digests = _relay_writer(tmp_path)
    intervals = boundary_stream.StreamedIntervals(source, head=source_head)
    interval = intervals[0]
    marker = {**intervals.consumed_markers()[0], name: bad}
    with pytest.raises(boundary_stream.BoundaryStreamError,
                       match=f"{name} does not bind exactly leads"):
        writer.write_segment(0, interval, relay_marker=marker)
    assert not boundary_stream.segment_marker_path(writer.root, 0).exists()


@pytest.mark.parametrize("start", [
    {"6": "bad"}, {"-1": "a" * 64}, {1.5: "a" * 64},
    {"6": "a" * 64, 6: "a" * 64}])
def test_a_tree_head_cannot_invent_a_start_marker_digest(tmp_path, start):
    with pytest.raises(ValueError, match="start-marker digest|each start lead"):
        _relay_writer(tmp_path, start=start)


def test_a_tree_head_takes_one_form_of_the_start_markers(tmp_path):
    with pytest.raises(ValueError, match="start markers.*never both"):
        _relay_writer(tmp_path, raw_start=True)


def test_a_relay_cannot_bind_another_interval_to_this_segments_records(tmp_path):
    writer, source, source_head, _digests = _relay_writer(tmp_path)
    intervals = boundary_stream.StreamedIntervals(source, head=source_head)
    interval = intervals[0]
    marker = {**intervals.consumed_markers()[0], "end_seconds": 7200.0}
    with pytest.raises(boundary_stream.BoundaryStreamError,
                       match="not the root interval this tree read"):
        writer.write_segment(0, interval, relay_marker=marker)
    assert not boundary_stream.segment_marker_path(writer.root, 0).exists()
