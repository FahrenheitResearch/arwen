"""Fresh-process forcing continuation keeps the checkpoint's bytes."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
import sys
import threading
import time

import pytest

from gpuwm.ingest import boundary_stream as stream
from gpuwm.ingest.stream_resume import (
    CONTINUATION_ENV, DESCRIPTOR_NAME, POSTED_PREFIX_DIRNAME,
    continue_prefix, preserve_posted_marker, seal_prefix, source_descriptor, verify_prefix,
)
from test_boundary_stream import (
    PROOF_HEAD, _assert_same_interval, _chained_tree, _frames, _initial, _met,
    _one_shot, _snapshots, _times,
)


def _prefix(tmp_path, *, seconds=60, required=None):
    writer, original = _chained_tree(tmp_path, _snapshots(), name="original", stop_after=0)
    prefix = tmp_path / "prefix"
    seal_prefix(original, seconds, prefix, required_intervals=required)
    return writer, original, prefix


def test_prefix_only_copies_committed_root_arrays_and_needed_interval(tmp_path):
    writer, original, prefix = _prefix(tmp_path)
    cache = original / "prepared-cache"
    (cache / "a99999.npy").write_bytes(b"part of an unpublished future interval")
    assert not (prefix / "prepared-cache" / "a99999.npy").exists()
    descriptor, head = verify_prefix(prefix)
    assert descriptor["prefix_intervals"] == 1
    assert descriptor["prefix_end_seconds"] == 3600
    assert not (prefix / "proof.json").exists()
    assert not (prefix / "prepared-cache/header.json").exists()
    assert not (prefix / "boundary-stream/producer.json").exists()
    one = stream.StreamedIntervals(original, head=head)[0]
    two = stream.StreamedIntervals(prefix, head=head)[0]
    _assert_same_interval(two, one)
    writer._stop_heartbeat(final="stopped")


def test_checkpoint_declared_prefix_cannot_be_truncated_to_its_clock(tmp_path):
    writer, original = _chained_tree(tmp_path, _snapshots(), name="original", stop_after=1)
    descriptor = seal_prefix(original, 60, tmp_path / "prefix", required_intervals=2)
    assert descriptor["prefix_intervals"] == 2
    assert descriptor["prefix_end_seconds"] == 7200
    writer._stop_heartbeat(final="stopped")


def test_checkpoint_at_a_forcing_boundary_seals_its_next_interval(tmp_path):
    writer, original = _chained_tree(tmp_path, _snapshots(), name="original", stop_after=1)
    descriptor = seal_prefix(original, 3600, tmp_path / "prefix")
    assert descriptor["prefix_intervals"] == 2
    assert descriptor["prefix_end_seconds"] == 7200
    writer._stop_heartbeat(final="stopped")


def test_boundary_checkpoint_defers_when_its_next_interval_is_missing(tmp_path):
    writer, original = _chained_tree(tmp_path, _snapshots(), stop_after=0)
    with pytest.raises(stream.BoundaryStreamError, match="interval 1 is not published"):
        seal_prefix(original, 3600, tmp_path / "prefix")
    assert not (tmp_path / "prefix").exists()
    writer._stop_heartbeat(final="stopped")


def test_resumed_prefix_can_publish_a_new_generation(tmp_path):
    writer, original, prefix = _prefix(tmp_path)
    next_prefix = tmp_path / "next-prefix"
    descriptor = seal_prefix(prefix, 120, next_prefix)
    assert DESCRIPTOR_NAME not in descriptor["files"]
    verify_prefix(next_prefix)
    writer._stop_heartbeat(final="stopped")


def test_final_checkpoint_uses_the_full_schedule_without_asking_for_an_extra_interval(tmp_path):
    writer, original = _chained_tree(tmp_path, _snapshots(), name="original")
    descriptor = seal_prefix(original, 10800, tmp_path / "final-prefix")
    assert descriptor["prefix_intervals"] == 3
    assert descriptor["prefix_end_seconds"] == 10800
    verify_prefix(tmp_path / "final-prefix")
    writer._stop_heartbeat(final="stopped")


@pytest.mark.parametrize("seconds,reason", [(3601, "interval 1 is not published"), (10801, "ends before")])
def test_missing_forcing_refuses_before_prefix_creation(tmp_path, seconds, reason):
    writer, original = _chained_tree(tmp_path, _snapshots(), stop_after=0)
    output = tmp_path / "refused"
    with pytest.raises(stream.BoundaryStreamError, match=reason):
        seal_prefix(original, seconds, output)
    assert not output.exists()
    writer._stop_heartbeat(final="stopped")


def test_corrupt_prefix_descriptor_or_array_refuses(tmp_path):
    writer, _, prefix = _prefix(tmp_path)
    descriptor = json.loads((prefix / DESCRIPTOR_NAME).read_text())
    array = next(name for name in descriptor["files"] if name.endswith(".npy"))
    (prefix / array).write_bytes(b"corrupt")
    with pytest.raises(stream.BoundaryStreamError, match="fails verification"):
        verify_prefix(prefix)
    descriptor["checkpoint_seconds"] = 1234
    (prefix / DESCRIPTOR_NAME).write_text(json.dumps(descriptor))
    with pytest.raises(stream.BoundaryStreamError, match="descriptor fails its digest"):
        verify_prefix(prefix)
    writer._stop_heartbeat(final="stopped")


def test_verified_replay_arrays_share_the_one_immutable_prefix_payload(tmp_path):
    import hashlib
    from gpuwm.ingest.stream_resume import _copy_array
    import numpy as np
    source, held = tmp_path / "replay/prepared-cache", tmp_path / "held/prepared-cache"
    source.mkdir(parents=True)
    held.mkdir(parents=True)
    value = np.arange(12, dtype=np.float32).reshape(3, 4)
    for directory in (source, held):
        np.save(directory / "a00000.npy", value, allow_pickle=False)
    from gpuwm.ingest.prepared_cache import _array_sha256
    spec = {"file": "a00000.npy", "shape": [3, 4], "dtype": "float32",
            "nbytes": value.nbytes, "sha256": _array_sha256(value)}
    path, twin = source / spec["file"], held / spec["file"]
    before = twin.read_bytes()
    assert not os.path.samefile(path, twin)
    _copy_array(source, held, "test/value", spec, share_existing=True)
    assert os.path.samefile(path, twin)
    assert twin.read_bytes() == path.read_bytes() == before
    assert not list(source.parent.glob("*.continuation-share"))


def test_producer_discovery_waits_for_the_real_hidden_atomic_tree_publish(tmp_path):
    from gpuwm.ingest.stream_resume import _find_producer_root
    from gpuwm.native_domain_artifacts import _atomic_staging_sibling
    producer = tmp_path / ".continuation"
    producer.mkdir()
    published = producer / "prepared"
    staging = _atomic_staging_sibling(published)
    staging.mkdir()
    observed = []

    def publish(source, target):
        head = stream.read_head(source)
        # This is the real writer's window between publishing head.json
        # in its private staging directory and the atomic directory rename.
        assert _find_producer_root(producer, head["head_sha256"]) == (None, None)
        observed.append(head["head_sha256"])
        os.replace(source, target)

    writer = stream.PreparedTreeWriter(staging=staging, output_root=published,
        identity={"source": "boundary-stream-test"}, chained=True, publish=publish)
    frames = _frames(_snapshots())
    try:
        assert writer.chained
        writer.write_head(initial_result=_initial(), met=_met(), forcing=frames,
            lbc={"spec_bdy_width": 5, "spec_zone": 1, "relax_zone": 4,
                 "schedule": [[0., 3600.], [3600., 7200.], [7200., 10800.]],
                 "fields": frames.inventory},
            proof_head=PROOF_HEAD, input_manifest_sha256="0" * 64)
        found, head = _find_producer_root(producer, writer.head_sha256)
        assert found == published
        assert head["head_sha256"] == writer.head_sha256
        assert observed == [writer.head_sha256]
        assert not staging.exists()
    finally:
        writer._stop_heartbeat(final="stopped")


def test_fresh_process_relay_publishes_future_intervals_before_its_seal(tmp_path):
    writer, _, prefix = _prefix(tmp_path)
    before = {name: (prefix / name).read_bytes() for name in verify_prefix(prefix)[0]["files"]}
    fresh = tmp_path / "fresh"
    script = tmp_path / "produce.py"
    script.write_text('''
from pathlib import Path
import sys,time
sys.path.insert(0, str(Path(sys.argv[2]).parent)); sys.path.insert(0, sys.argv[2])
from test_boundary_stream import _chained_tree,_snapshots,_frames,_times,PROOF_HEAD
pack=_snapshots()
writer,root=_chained_tree(Path(sys.argv[1]),pack,name="fresh",stop_after=0)
frames=_frames(pack)
for index in (1,2):
    writer.write_segment(index,frames.interval(index,_times(len(pack))))
    acknowledged=Path(sys.argv[1])/f"consumer-marker-{index}.ack"
    deadline=time.monotonic()+60
    while not acknowledged.is_file():
        if time.monotonic()>=deadline:
            raise TimeoutError(f"consumer did not acknowledge interval {index}")
        time.sleep(.05)
receipt=writer.seal_cache()
writer.publish({**PROOF_HEAD,"prepared_cache":{"content_sha256":receipt["content_sha256"]},
                "boundary_stream":writer.boundary_stream_proof()})
''', encoding="utf-8")
    seen = []
    def segment(index):
        assert not (fresh / "proof.json").exists()
        assert stream.segment_marker_path(prefix, index).is_file()
        seen.append(index)
        (tmp_path / f"consumer-marker-{index}.ack").write_text(str(index))
    result = continue_prefix(prefix, fresh, producer_argv=[sys.executable, str(script),
                             str(tmp_path), str(Path(__file__).parent)],
                             poll_seconds=.025, timeout_seconds=120, on_segment=segment)
    assert seen == [1, 2]
    assert result["status"] == "PASS"
    assert result["continued_intervals"] == 2
    assert all((prefix / name).read_bytes() == value for name, value in before.items())
    head = stream.read_head(prefix)
    left, right = stream.StreamedIntervals(prefix, head=head), stream.StreamedIntervals(fresh, head=head)
    for index in range(3):
        _assert_same_interval(left[index], right[index])
    writer._stop_heartbeat(final="stopped")


def test_relay_refuses_changed_prefix_interval_and_keeps_old_bytes(tmp_path):
    writer, _, prefix = _prefix(tmp_path)
    before = stream.segment_marker_path(prefix, 0).read_bytes()
    other, source = _chained_tree(tmp_path, _snapshots(seed=1), name="changed")
    with pytest.raises(stream.BoundaryStreamError, match="changes sealed interval"):
        continue_prefix(prefix, source, poll_seconds=.01, timeout_seconds=1)
    assert stream.segment_marker_path(prefix, 0).read_bytes() == before
    assert (prefix / "boundary-stream/failed.json").is_file()
    writer._stop_heartbeat(final="stopped")
    other._stop_heartbeat(final="stopped")


def test_a_dead_producer_is_a_clean_stream_failure(tmp_path):
    writer, _, prefix = _prefix(tmp_path)
    with pytest.raises(stream.BoundaryProducerFailed, match="exited 7"):
        continue_prefix(prefix, tmp_path / "absent", producer_argv=[sys.executable, "-c", "raise SystemExit(7)"],
                        poll_seconds=.01, timeout_seconds=5)
    assert "exited 7" in (prefix / "boundary-stream/failed.json").read_text()
    writer._stop_heartbeat(final="stopped")


def _marker():
    return {"schema": stream.POSTED_LEAD_SCHEMA, "source": "gfs", "member": None,
            "cycle": "2026-10-04T00:00:00Z", "lead": 1, "valid_time": "2026-10-04T01:00:00Z",
            "objects": [{"name": "f001.grib2", "bytes": 20, "sha256": "a" * 64,
                         "endpoint": "first"}], "fetched_at": "old"}


def test_prefix_provenance_reuses_times_only_after_reverifying_source_bytes(tmp_path, monkeypatch):
    held = _marker()
    path = stream.stream_dir(tmp_path) / POSTED_PREFIX_DIRNAME / "f001.json"
    stream._write_json_atomic(path, held)
    monkeypatch.setenv(CONTINUATION_ENV, str(tmp_path))
    candidate = json.loads(json.dumps(held))
    candidate["fetched_at"] = "new"
    candidate["objects"][0]["endpoint"] = "second"
    assert preserve_posted_marker(candidate) == held
    candidate["objects"][0]["sha256"] = "b" * 64
    with pytest.raises(stream.BoundaryStreamError, match="changed sealed lead 1"):
        preserve_posted_marker(candidate)


def test_prepare_only_go_parser_supports_the_continuation_command():
    from gpuwm.cli import build_parser
    args = build_parser().parse_args(["go", "frozen.toml", "--prepare-only", "--run-stamp", "off"])
    assert args.prepare_only


def test_portable_generic_authorities_are_hash_bound_with_stem_companions(tmp_path):
    import hashlib
    config = tmp_path / "experiment.toml"
    config.write_text('[fetch]\nsource="gfs"\ncycle="2026-10-04T00"\n')
    for name in ("namelist.wps", "namelist.input", "stock.namelist.input", "d01-target.json",
                 "experiment.namelist.wps"):
        (tmp_path / name).write_bytes(name.encode())
    descriptor = source_descriptor(config)
    assert descriptor["config_name"] == "experiment.toml"
    assert descriptor["cycle"] == "2026-10-04T00"
    assert set(descriptor["authorities"]) == {
        "experiment.toml", "namelist.wps", "namelist.input", "stock.namelist.input",
        "d01-target.json", "experiment.namelist.wps"}
    for name, digest in descriptor["authorities"].items():
        assert digest == hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    config.write_text('[fetch]\nsource="gfs"\ncycle="latest"\n')
    with pytest.raises(stream.BoundaryStreamError, match="frozen source and cycle"):
        source_descriptor(config)


@pytest.mark.skipif(os.name == "nt", reason="production producer leases use Linux flock")
def test_distinct_prefix_helpers_hold_only_one_source_producer_lease(tmp_path, monkeypatch):
    from gpuwm.ingest.stream_resume import PRODUCER_LOCK_ENV
    trace = tmp_path / "producer-order.txt"
    script = tmp_path / "leased-producer.py"
    script.write_text('''
from pathlib import Path
import sys,time
sys.path.insert(0,str(Path(sys.argv[2]).parent)); sys.path.insert(0,sys.argv[2])
from test_boundary_stream import _chained_tree,_snapshots
trace=Path(sys.argv[3]); tag=sys.argv[4]
with trace.open("a") as out: out.write(tag+":start\\n")
time.sleep(.5)
_chained_tree(Path(sys.argv[1]),_snapshots(),name="fresh")
with trace.open("a") as out: out.write(tag+":end\\n")
''', encoding="utf-8")
    monkeypatch.setenv(PRODUCER_LOCK_ENV, str(tmp_path / "producer.lock"))
    writers, tasks, errors = [], [], []
    for name in ("one", "two"):
        folder = tmp_path / name
        folder.mkdir()
        writer, _, prefix = _prefix(folder)
        writers.append(writer)
        def run(folder=folder, prefix=prefix, name=name):
            try:
                continue_prefix(prefix, folder / "fresh", producer_argv=[sys.executable, str(script),
                    str(folder), str(Path(__file__).parent), str(trace), name],
                    poll_seconds=.025, timeout_seconds=60)
            except BaseException as error:
                errors.append(error)
        tasks.append(threading.Thread(target=run))
    for task in tasks:
        task.start()
    for task in tasks:
        task.join(timeout=120)
    assert not errors
    assert all(not task.is_alive() for task in tasks)
    lines = trace.read_text().splitlines()
    first = lines[0].split(":")[0]
    other = "two" if first == "one" else "one"
    assert lines == [first+":start", first+":end", other+":start", other+":end"]
    for writer in writers:
        writer._stop_heartbeat(final="stopped")


def _nested_metadata_source(base, *, name, workers=2, stamp="2026-10-04T00:00:00Z",
                            stop_after=None, mutate=None):
    import hashlib
    import numpy as np
    from gpuwm.ingest.boundary_stream import PreparedTreeWriter, domain_tree_head_fields
    from gpuwm.ingest.lateral_bc import StateBoundaryFrames
    from gpuwm.ingest.preprocess_backend import PREPROCESS_IMPLEMENTATION_SCHEMA, preprocess_identity
    from gpuwm.native_wrf_contract import write_native_static_cache
    snapshots = _snapshots()
    staging = base / (".tmp-" + name)
    staging.mkdir()
    child = staging / "hierarchy-head/domains/d02"
    child.mkdir(parents=True)
    child_build = base / (name + "-child-build")
    child_build.mkdir()
    cache, receipt, _, _ = _one_shot(child_build, snapshots)
    shutil.copytree(cache, child / "prepared-cache")
    header_path = child / "prepared-cache/header.json"
    header = json.loads(header_path.read_text())
    header["created_utc"] = stamp
    if mutate == "child-header":
        header["metadata"]["user"]["changed-science"] = True
    header_path.write_text(json.dumps(header, sort_keys=True))
    static = write_native_static_cache(child / "native-static.npz", {"field": np.arange(4)})
    (child / "geometry-receipt.json").write_text(json.dumps({"cache": static, "geometry": {"nx": 2}}))
    child_receipt = {"schema": "gpuwm-native-domain-artifact-build-v1", "grid_id": 2,
        "parent_id": 1, "boundary_mode": "nested-parent-forced", "valid_time": "2026-10-04T00:00:00",
        "artifacts": {"prepared_cache": {"content_sha256": receipt["content_sha256"],
            "header_sha256": hashlib.sha256(header_path.read_bytes()).hexdigest()}, "static_cache": static},
        "input_preparation_seconds": float(workers)}
    child_receipt["preprocess_receipt"] = {
        "backend": "cpu", "workers": workers,
        "soil": {"preprocess_backend": {"backend": "cpu", "workers": workers,
            "bridge": {"sha256": "c" * 64}}}}
    if mutate == "child-receipt":
        child_receipt["undeclared-science"] = True
    if mutate == "child-implementation":
        child_receipt["preprocess_receipt"]["soil"]["preprocess_backend"]["bridge"]["sha256"] = "d" * 64
    (child / "receipt.json").write_text(json.dumps(child_receipt, sort_keys=True))
    if mutate == "child-bytes":
        payload = next((child / "prepared-cache").glob("*.npy"))
        payload.write_bytes(b"changed child payload")
    if mutate == "extra-payload":
        (child / "unknown-science.npy").write_bytes(b"additional scientific payload")
    preprocessing = {"schema": PREPROCESS_IMPLEMENTATION_SCHEMA, "backend": "cpu",
        "implementation": "fixed-test-native-bridge", "bridge": {"sha256": "a" * 64},
        "workers": workers, "host_cpu_count": workers * 2, "parallelism": {"effective": workers},
        "selection": {"backend": "cpu", "host_fit": {"available_bytes": workers * 1000}}}
    if mutate == "implementation":
        preprocessing["bridge"]["sha256"] = "b" * 64
    proof_head = {**PROOF_HEAD, "preprocessing": preprocessing,
        "preprocessing_receipt_sha256": stream._canonical(preprocessing), "hierarchy_workers": workers}
    if mutate == "head-field":
        proof_head["undeclared-science"] = True
    proof_head["preprocessing_receipt_sha256"] = hashlib.sha256(
        proof_head["preprocessing_receipt_sha256"].encode()).hexdigest()
    root_cache = "hierarchy-head/domains/d01/prepared-cache"
    writer = PreparedTreeWriter(staging=staging, output_root=base / name,
        identity={"source": "metadata-replay-test"}, cache_name=root_cache, chained=True)
    frames = StateBoundaryFrames(spec_bdy_width=5, spec_zone=1, relax_zone=4)
    frames.add_snapshot(snapshots[0], index=0)
    initial = _initial()
    if mutate == "root-bytes":
        initial.state.u += 1
    seconds = [0., 3600., 7200., 10800.]
    writer.write_head(initial_result=initial, met=_met(), forcing=frames,
        lbc={"spec_bdy_width": 5, "spec_zone": 1, "relax_zone": 4,
             "schedule": list(map(list, zip(seconds, seconds[1:]))), "fields": frames.inventory},
        proof_head=proof_head, input_manifest_sha256="0" * 64,
        tree=domain_tree_head_fields(["d01", "d02"], root_cache=root_cache,
            children_receipts={"d02": hashlib.sha256((child / "receipt.json").read_bytes()).hexdigest()}))
    for index in range(3):
        frames.add_snapshot(snapshots[index + 1], index=index + 1)
        writer.write_segment(index, frames.interval(index, _times(4)))
        frames.release(index)
        if stop_after == index:
            return writer, writer.root
    cache_receipt = writer.seal_cache()
    writer.publish({**proof_head, "prepared_cache": {"content_sha256": cache_receipt["content_sha256"]},
                    "boundary_stream": writer.boundary_stream_proof()})
    return writer, writer.root


def test_real_child_header_times_and_declared_root_telemetry_keep_the_original_head_and_seal(tmp_path, monkeypatch):
    old, original = _nested_metadata_source(tmp_path, name="original", stop_after=0)
    prefix = tmp_path / "prefix"
    seal_prefix(original, 60, prefix)
    source = tmp_path / "fresh"
    script = tmp_path / "fresh-nested-metadata.py"
    script.write_text('''
from pathlib import Path
import sys
sys.path.insert(0,str(Path(sys.argv[2]).parent)); sys.path.insert(0,sys.argv[2])
from test_stream_resume import _nested_metadata_source
_nested_metadata_source(Path(sys.argv[1]),name="fresh",workers=8,stamp="2026-10-04T02:00:00Z")
''', encoding="utf-8")
    result = continue_prefix(prefix, source, producer_argv=[sys.executable, str(script),
        str(tmp_path), str(Path(__file__).parent)], timeout_seconds=120)
    assert stream.read_head(source)["head_sha256"] == stream.read_head(prefix)["head_sha256"]
    child = Path("hierarchy-head/domains/d02")
    assert (source / child / "prepared-cache/header.json").read_bytes() == (prefix / child / "prepared-cache/header.json").read_bytes()
    assert (source / child / "receipt.json").read_bytes() == (prefix / child / "receipt.json").read_bytes()
    receipt = json.loads((source / "boundary-stream/continuation-preparation.json").read_text())
    assert receipt["fresh_producer"]["preprocessing"]["workers"] == 8
    assert result["status"] == "PASS"
    stream.verify_seal(prefix, head=stream.read_head(prefix))
    old._stop_heartbeat(final="stopped")


@pytest.mark.parametrize("mutation", ["child-header", "child-bytes", "child-receipt",
    "child-implementation", "extra-payload", "head-field", "implementation", "root-bytes"])
def test_replay_metadata_never_hides_changed_scientific_data(tmp_path, monkeypatch, mutation):
    old, original = _nested_metadata_source(tmp_path, name="original", stop_after=0)
    prefix = tmp_path / "prefix"
    seal_prefix(original, 60, prefix)
    before = (prefix / "boundary-stream/head.json").read_bytes()
    monkeypatch.setenv(CONTINUATION_ENV, str(prefix))
    with pytest.raises(stream.BoundaryStreamError, match="continuation changes"):
        _nested_metadata_source(tmp_path, name="changed", workers=8, mutate=mutation)
    assert (prefix / "boundary-stream/head.json").read_bytes() == before
    old._stop_heartbeat(final="stopped")
