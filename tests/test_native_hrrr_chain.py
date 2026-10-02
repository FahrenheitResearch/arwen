"""A136 L7c: a single native HRRR domain prepares chained.

The native preparation publishes the portable bundle's head once its start
state exists, one boundary interval per hour after it, and ``proof.json``
at its seal.  Two things differ from the routes that chained before it,
and each is pinned here: the cache's per-lead ``mapping_reports`` exist
only once each lead is mapped, so the head holds the start lead's and the
seal completes the rest; and the preparation writes into a bundle root
that already exists, so the head is published in place.
"""

from __future__ import annotations

import errno
import gc
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.ingest import boundary_stream
from gpuwm.ingest.boundary_stream import (
    BoundaryStreamError, PreparedTreeWriter, read_head, verify_seal,
)
from gpuwm.ingest.prepared_cache import (
    SEAL_COMPLETES_KEY, PreparedCacheReader, write_prepared_cache,
)
from gpuwm.ingest.lateral_bc import StateBoundaryFrames

from test_boundary_stream import _frames, _initial, _met, _snapshots, _times


IDENTITY = {"source": "native-hrrr-chain-test"}
PROOF_HEAD = {"schema": "test-proof", "forcing_hours": [0, 1, 2, 3]}


def _reports(count):
    return {f"f{hour:02d}": {"policy": f"lead {hour}", "sides": {
        "west": {"soil": hour}}} for hour in range(count)}


def _one_shot(tmp_path, snapshots):
    times = _times(len(snapshots))
    boundaries = _frames(snapshots).build(times)
    path = tmp_path / "one-shot"
    receipt = write_prepared_cache(
        path, identity=IDENTITY, initial_result=_initial(boundaries),
        met=_met(), boundaries=boundaries,
        metadata={"mapping_reports": _reports(len(snapshots)), "cycle": 6})
    return path, receipt


def _in_place(tmp_path, snapshots, *, completed=None, chained=True):
    """The native layout: the bundle root exists, the head is written in it."""

    times = _times(len(snapshots))
    root = tmp_path / "bundle"
    (root / "native").mkdir(parents=True)

    def in_place(staging, output):
        assert Path(staging) == Path(output)

    writer = PreparedTreeWriter(
        staging=root, output_root=root, identity=IDENTITY, chained=chained,
        cache_name="native/prepared-cache", publish=in_place)
    frames = StateBoundaryFrames(spec_bdy_width=5, spec_zone=1, relax_zone=4)
    frames.add_snapshot(snapshots[0], index=0)
    seconds = [(t - times[0]).total_seconds() for t in times]
    writer.write_head(
        initial_result=_initial(), met=_met(), lbc={
            "spec_bdy_width": 5, "spec_zone": 1, "relax_zone": 4,
            "schedule": [[seconds[k], seconds[k + 1]]
                         for k in range(len(times) - 1)],
            "fields": frames.inventory},
        metadata={"mapping_reports": _reports(1), "cycle": 6},
        proof_head=PROOF_HEAD, input_manifest_sha256="0" * 64,
        forcing=frames, seal_completes=("mapping_reports",))
    for index in range(1, len(snapshots)):
        frames.add_snapshot(snapshots[index], index=index)
        writer.write_segment(index - 1, frames.interval(index - 1, times))
        frames.release(index - 1)
    receipt = writer.seal_cache(completed_metadata=(
        {"mapping_reports": _reports(len(snapshots))}
        if completed is None else completed))
    writer.publish({**PROOF_HEAD,
                    "prepared_cache": {"content_sha256":
                                       receipt["content_sha256"]},
                    "boundary_stream": writer.boundary_stream_proof()})
    return writer, root, receipt


def test_a_completed_seal_is_the_one_shot_cache(tmp_path):
    snapshots = _snapshots(4)
    path, one = _one_shot(tmp_path, snapshots)
    writer, root, two = _in_place(tmp_path, snapshots)
    assert two["content_sha256"] == one["content_sha256"]
    first = json.loads((path / "header.json").read_text(encoding="utf-8"))
    cache = root / "native" / "prepared-cache"
    second = json.loads((cache / "header.json").read_text(encoding="utf-8"))
    assert second["metadata"] == first["metadata"]
    assert second["arrays"] == first["arrays"]
    for spec in first["arrays"].values():
        assert (cache / spec["file"]).read_bytes() \
            == (path / spec["file"]).read_bytes()
    head = read_head(root)
    # The head held the start lead's report only, and named the key.
    assert head["basis"]["cache"][SEAL_COMPLETES_KEY] == ["mapping_reports"]
    assert list(head["basis"]["cache"]["metadata"]["user"][
        "mapping_reports"]) == ["f00"]
    assert verify_seal(root, head=head)["content_sha256"] \
        == one["content_sha256"]
    assert PreparedCacheReader(cache, expected_identity=IDENTITY
                               ).verify_all()["content_sha256"] \
        == one["content_sha256"]


def test_a_head_that_names_nothing_is_the_head_it_always_was(tmp_path):
    writer = PreparedTreeWriter(
        staging=tmp_path / ".s", output_root=tmp_path / "o",
        identity=IDENTITY, chained=False)
    frames = StateBoundaryFrames(spec_bdy_width=5, spec_zone=1, relax_zone=4)
    frames.add_snapshot(_snapshots(2)[0], index=0)
    writer.write_head(
        initial_result=_initial(), met=_met(), lbc={
            "spec_bdy_width": 5, "spec_zone": 1, "relax_zone": 4,
            "schedule": [[0.0, 3600.0]], "fields": frames.inventory},
        proof_head=PROOF_HEAD, forcing=frames)
    assert SEAL_COMPLETES_KEY not in writer.head["basis"]["cache"]


@pytest.mark.parametrize("completed, words", [
    ({"cycle": 7}, "did not name"),
    ({"mapping_reports": {**_reports(4), "f00": {"policy": "other"}}},
     "changes or drops mapping_reports entries \\['f00'\\]"),
    ({"mapping_reports": {"f01": {"policy": "lead 1"}}},
     "changes or drops mapping_reports entries \\['f00'\\]"),
])
def test_a_seal_that_rewrites_what_its_head_wrote_is_refused(
        tmp_path, completed, words):
    with pytest.raises(ValueError, match=words):
        _in_place(tmp_path, _snapshots(3), completed=completed)


def test_a_head_names_only_a_mapping_it_holds(tmp_path):
    writer = PreparedTreeWriter(
        staging=tmp_path / ".s", output_root=tmp_path / "o",
        identity=IDENTITY, chained=False)
    frames = StateBoundaryFrames(spec_bdy_width=5, spec_zone=1, relax_zone=4)
    frames.add_snapshot(_snapshots(2)[0], index=0)
    with pytest.raises(ValueError, match="holds no mapping"):
        writer.write_head(
            initial_result=_initial(), met=_met(), lbc={
                "spec_bdy_width": 5, "spec_zone": 1, "relax_zone": 4,
                "schedule": [[0.0, 3600.0]], "fields": frames.inventory},
            metadata={"cycle": 6}, proof_head=PROOF_HEAD, forcing=frames,
            seal_completes=("mapping_reports",))


def test_the_seal_check_holds_every_other_user_key_and_every_head_entry():
    head = {"mapping_reports": {"f00": 1}, "cycle": 6}
    boundary_stream._require_completed_user(
        {"mapping_reports": {"f00": 1, "f01": 2}, "cycle": 6}, head,
        {"mapping_reports"})
    for sealed in ({"mapping_reports": {"f00": 9, "f01": 2}, "cycle": 6},
                   {"mapping_reports": {"f01": 2}, "cycle": 6},
                   {"mapping_reports": {"f00": 1}, "cycle": 7},
                   {"mapping_reports": {"f00": 1}}):
        with pytest.raises(BoundaryStreamError):
            boundary_stream._require_completed_user(
                sealed, head, {"mapping_reports"})


def test_the_lbc_digest_fed_in_order_is_the_whole_set_digest():
    from tools import hrrr_single_domain_benchmark as bench

    snapshots = _snapshots(4)
    times = _times(4)
    boundaries = _frames(snapshots).build(times)
    digest = bench._LbcPayloadDigest()
    for index, interval in enumerate(boundaries.intervals):
        digest.add(index, interval)
    assert digest.hexdigest() == bench._lbc_payload_sha256(boundaries)
    with pytest.raises(ValueError, match="out of order"):
        bench._LbcPayloadDigest().add(1, boundaries.intervals[1])


def test_the_interval_price_is_the_frame_price():
    from tools import hrrr_single_domain_benchmark as bench

    snapshots = _snapshots(2)
    frames = StateBoundaryFrames(spec_bdy_width=5, spec_zone=1, relax_zone=4)
    frames.add_snapshot(snapshots[0], index=0)
    sides = next(iter(frames._frames.values()))
    assert bench._interval_host_pricing(sides).interval_host_bytes \
        == frames.interval_host_bytes


def test_the_native_sealed_reason_is_retired_and_its_door_chains():
    assert "native_hrrr" not in boundary_stream.SEALED_REASONS
    root = Path(__file__).resolve().parents[1]
    bench = (root / "tools/hrrr_single_domain_benchmark.py").read_text(
        encoding="utf-8")
    assert 'say_prepared_sealed("native_hrrr")' not in bench
    from gpuwm import runplan

    chain = inspect.getsource(runplan._hrrr_chain)
    single = inspect.getsource(runplan._hrrr_single_chain)
    # One domain: the forecast binds the head the preparation publishes.
    assert "_hrrr_single_chain(" in chain
    assert "run_chained(" in single
    assert '"--prepared-head-sha256", head_sha256' in single
    # A native tree chains too (A136 L7c): the hierarchy stage starts on
    # the root preparation's head and the tree runner binds the tree's.
    assert 'say_prepared_sealed("domain_tree")' not in chain
    assert "domain_tree" not in boundary_stream.SEALED_REASONS
    tree = inspect.getsource(runplan._hrrr_tree_chain)
    assert "_hrrr_tree_chain(" in chain
    assert "run_chained(" in tree
    assert "fresh_chained_head(prep_root" in tree
    forecast = inspect.getsource(runplan._hrrr_tree_forecast)
    assert '"--prepared-head-sha256", str(head_sha256)' in forecast


def test_the_wrapper_chains_by_default_and_not_for_a_sealed_prefix(
        monkeypatch):
    import tools.prepare_hrrr_wrf as prepare

    monkeypatch.delenv(boundary_stream.CHAINED_ENV, raising=False)
    monkeypatch.setattr(boundary_stream, "forecast_installed", lambda: True)
    assert prepare._chains(SimpleNamespace(sealed_prepared_cache=False))
    assert not prepare._chains(SimpleNamespace(sealed_prepared_cache=True))
    monkeypatch.setenv(boundary_stream.CHAINED_ENV, "0")
    assert not prepare._chains(SimpleNamespace(sealed_prepared_cache=False))
    monkeypatch.delenv(boundary_stream.CHAINED_ENV)
    monkeypatch.setattr(boundary_stream, "forecast_installed", lambda: False)
    assert not prepare._chains(SimpleNamespace(sealed_prepared_cache=False))


def test_a_chained_bundle_is_refused_outside_a_new_pipeline_preparation(
        tmp_path):
    from tools import hrrr_single_domain_benchmark as bench

    document = tmp_path / "chain.json"
    document.write_text(json.dumps({"schema": bench.CHAINED_BUNDLE_SCHEMA}),
                        encoding="utf-8")
    base = dict(chained_bundle=document, prepare_only=True,
                pipeline_series=tmp_path / "series.tsv",
                sealed_prepared_cache=False)
    assert bench._chained_bundle(SimpleNamespace(**base))["schema"] \
        == bench.CHAINED_BUNDLE_SCHEMA
    assert bench._chained_bundle(
        SimpleNamespace(**{**base, "chained_bundle": None})) is None
    for change in ({"prepare_only": False}, {"pipeline_series": None},
                   {"sealed_prepared_cache": True}):
        with pytest.raises(ValueError, match="new prepare-only pipeline"):
            bench._chained_bundle(SimpleNamespace(**{**base, **change}))
    document.write_text(json.dumps({"schema": "other"}), encoding="utf-8")
    with pytest.raises(ValueError, match="is not a"):
        bench._chained_bundle(SimpleNamespace(**base))


def test_the_split_bundle_writer_is_the_one_shot_writer():
    from gpuwm import hrrr_prepared_bundle as bundle

    one_shot = inspect.getsource(bundle.publish_hrrr_prepared_bundle)
    # One code path: the one-shot publication is the head, then the seal.
    assert "publish_hrrr_bundle_head(" in one_shot
    assert "seal_hrrr_bundle_proof(" in one_shot
    assert "sealed_handoff(" in one_shot
    # Every key the seal adds is a seal-only proof key, so the sealed
    # proof is the head proof plus exactly those.
    seal = inspect.getsource(bundle.seal_hrrr_bundle_proof)
    for key in ("initialization_artifacts", "prepared_cache", "export"):
        assert f'"{key}"' in seal
        assert key in boundary_stream.SEAL_ONLY_PROOF_KEYS


_CIMIXR_GATE = ("PASS discipline=0 category=1 parameter=82 "
                "level_type=105; finite/nonnegative/nonzero")


def _long_bridge(root, hours):
    """A sealed native bridge of ``hours`` leads, one 1x1 column each."""

    import hashlib

    from gpuwm.ingest import hrrr

    root.mkdir()
    (root / "gate.txt").write_text(
        "status\tPASS\n"
        "cycle\t2026-07-18 00:00:00\n"
        f"forecast_hours\t{','.join(map(str, hours))}\n"
        f"series_count\t{len(hours)}\n"
        "atmosphere_selected_per_time\t561\n"
        "hybrid_levels\t50\n"
        "soil_selected_per_time\t18\n"
        "window_zero_based_inclusive\ti=10..10 j=20..20\n"
        "window_shape\t1x1\n"
        f"qice_mapping\t{_CIMIXR_GATE}\n"
        "cross_time_inventory\tPASS exact selected keys/levels/grid\n")
    for hour in hours:
        atmosphere = root / f"atmosphere-f{hour:02d}"
        soil = root / f"soil-f{hour:02d}"
        atmosphere.mkdir()
        soil.mkdir()
        for name in hrrr._ATMOSPHERE_3D:
            np.full((50, 1, 1), hour, dtype="<f4").tofile(
                atmosphere / f"{name}.f32le")
        for name in hrrr._ATMOSPHERE_2D:
            np.full((1, 1), hour, dtype="<f4").tofile(
                atmosphere / f"{name}.f32le")
        for name in hrrr._SOIL_3D:
            np.full((9, 1, 1), hour, dtype="<f4").tofile(
                soil / f"{name}.f32le")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    lines = [f"{digest(path)}  ./{path.relative_to(root).as_posix()}"
             for path in sorted(root.rglob("*")) if path.is_file()]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    return digest(root / "SHA256SUMS")


def _open_descriptors():
    return len(os.listdir("/proc/self/fd"))


@pytest.mark.skipif(not Path("/proc/self/fd").is_dir(),
                    reason="counts descriptors through /proc/self/fd")
def test_the_head_maps_a_48_hour_bridge_one_lead_at_a_time(tmp_path):
    """The chained head reads every later lead from the sealed bridge.

    It mapped all of them at once, and a mapped lead holds one descriptor
    per field, so a 48 h native preparation died with ``[Errno 24] Too
    many open files`` under the ordinary 1024 soft limit before its head.
    The head's leads are now mapped as each hour is taken, which is what
    the one-shot pipeline route holds.
    """

    resource = pytest.importorskip("resource")
    from tools import hrrr_single_domain_benchmark as bench

    from gpuwm.ingest.hrrr import load_hrrr_native_series

    hours = tuple(range(49))
    later = hours[1:]
    manifest = _long_bridge(tmp_path / "bridge", hours)
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    limit = 1024 if hard == resource.RLIM_INFINITY else min(1024, hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (limit, hard))
    try:
        baseline = _open_descriptors()
        leads = bench._SealedBridgeLeads()
        assert not leads
        leads.open(tmp_path / "bridge", {hour: hour for hour in later},
                   expected_manifest_sha256=manifest)
        assert leads
        per_lead = 0
        for hour in later:
            snapshot = leads.take(hour)
            assert snapshot.forecast_hour == hour
            assert float(snapshot.fields["TT"][0, 0, 0]) == float(hour)
            held = _open_descriptors() - baseline
            per_lead = max(per_lead, held)
            # One lead's fields and nothing carried from the leads before.
            assert held <= 3 * 24, (hour, held)
            del snapshot
        # Each hour is taken once, as the head's dict pop was.
        with pytest.raises(KeyError):
            leads.take(later[-1])
        if baseline + per_lead * len(later) > limit:
            # The same window mapped at once, as the head did, still
            # fails here, so this test can tell the two apart.
            failed = None
            try:
                load_hrrr_native_series(
                    tmp_path / "bridge", later,
                    expected_manifest_sha256=manifest)
            except OSError as error:
                failed = error.errno
            gc.collect()
            assert failed == errno.EMFILE
    finally:
        gc.collect()
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


def test_the_chained_head_and_run_take_leads_from_the_verified_bridge():
    from tools import hrrr_single_domain_benchmark as bench

    head = inspect.getsource(bench._write_chained_head)
    assert "load_hrrr_native_series" not in head
    assert "sealed_leads.open(" in head
    run = inspect.getsource(bench.run)
    assert "load_hrrr_native_series" not in run
    assert run.count("sealed_leads.take(hour)") == 2


# ---------------------------------------------------------------------------
# A native HRRR domain tree chains on its root preparation's head
# ---------------------------------------------------------------------------


def _tree_writer(tmp_path, snapshots, *, proof_name, name, stop_after=None):
    """A tree-shaped writer whose seal writes ``proof_name``."""

    times = _times(len(snapshots))
    staging = tmp_path / f".tmp-{name}"
    staging.mkdir()
    writer = PreparedTreeWriter(
        staging=staging, output_root=tmp_path / name, identity=IDENTITY,
        chained=True, proof_name=proof_name)
    frames = StateBoundaryFrames(spec_bdy_width=5, spec_zone=1, relax_zone=4)
    frames.add_snapshot(snapshots[0], index=0)
    seconds = [(t - times[0]).total_seconds() for t in times]
    writer.write_head(
        initial_result=_initial(), met=_met(), lbc={
            "spec_bdy_width": 5, "spec_zone": 1, "relax_zone": 4,
            "schedule": [[seconds[k], seconds[k + 1]]
                         for k in range(len(times) - 1)],
            "fields": frames.inventory},
        metadata={"root_preparation": {}}, proof_head=PROOF_HEAD,
        input_manifest_sha256="0" * 64, forcing=frames,
        seal_completes=("root_preparation",))
    for index in range(1, len(snapshots)):
        frames.add_snapshot(snapshots[index], index=index)
        writer.write_segment(index - 1, frames.interval(index - 1, times))
        frames.release(index - 1)
        if stop_after is not None and index - 1 == stop_after:
            return writer
    return writer


def _seal(writer, digest="a" * 64):
    receipt = writer.seal_cache(completed_metadata={
        "root_preparation": {"prepared_content_sha256": digest}})
    writer.publish({**PROOF_HEAD,
                    "root_preparation": {"prepared_content_sha256": digest},
                    "prepared_cache": {"content_sha256":
                                       receipt["content_sha256"]},
                    "boundary_stream": writer.boundary_stream_proof()})
    return receipt


def test_a_tree_head_names_its_receipt_and_is_complete_only_when_sealed(
        tmp_path):
    """The native tree seals receipt.json, as its one-shot tree always has.

    The head names that document in its basis (bound by the head digest);
    completion, the consumer's seal wait and verify_seal read it, and a
    head that seals proof.json names nothing, so every other head's digest
    is what it was.
    """

    snapshots = _snapshots(4)
    writer = _tree_writer(tmp_path, snapshots, proof_name="receipt.json",
                          name="tree")
    root = tmp_path / "tree"
    head = read_head(root)
    assert head["basis"]["proof_name"] == "receipt.json"
    assert boundary_stream.proof_document_name(head) == "receipt.json"
    assert not boundary_stream.prepared_tree_complete(root)
    assert not boundary_stream.StreamedIntervals(root, head=head).sealed()
    receipt = _seal(writer)
    assert (root / "receipt.json").is_file()
    assert not (root / "proof.json").exists()
    assert boundary_stream.prepared_tree_complete(root)
    sealed = verify_seal(root, head=head)
    assert sealed["content_sha256"] == receipt["content_sha256"]
    # The seal completed the root preparation's record the head held empty.
    header = json.loads((root / "prepared-cache" / "header.json").read_text(
        encoding="utf-8"))
    assert header["metadata"]["user"]["root_preparation"] == {
        "prepared_content_sha256": "a" * 64}

    plain = _tree_writer(tmp_path, snapshots, proof_name="proof.json",
                         name="plain", stop_after=0)
    assert "proof_name" not in plain.head["basis"]
    assert boundary_stream.proof_document_name(plain.head) == "proof.json"


def test_root_preparation_is_a_seal_only_key_and_the_head_holds_it_out(
        tmp_path):
    assert "root_preparation" in boundary_stream.SEAL_ONLY_PROOF_KEYS
    with pytest.raises(ValueError, match="seal-only"):
        PreparedTreeWriter(
            staging=tmp_path / ".x", output_root=tmp_path / "x",
            identity=IDENTITY, chained=False).write_head(
                initial_result=_initial(), met=_met(), lbc=None,
                proof_head={"root_preparation": {}})


def test_the_tree_runner_binds_a_head_by_the_document_it_names(
        tmp_path, monkeypatch):
    """A native HRRR tree's head is matched to its receipt.json schema."""

    from gpuwm import prepared_domain_tree_forecast as runner

    head = {"head_sha256": "h" * 64, "basis": {
        "proof_name": "receipt.json",
        "tree": {"layout": boundary_stream.LAYOUT_DOMAIN_TREE},
        "proof_head": {"schema": runner.HIERARCHY_SCHEMA, "status": "PASS"}}}
    monkeypatch.setattr(boundary_stream, "bind_head", lambda *a, **k: head)
    _path, document, source, bound = runner._load_head_document(
        tmp_path, "h" * 64)
    assert source == "hrrr" and document["schema"] == runner.HIERARCHY_SCHEMA
    # The same document named as proof.json is no hierarchy this runner
    # reads: the head's name and the schema's file must agree.
    del head["basis"]["proof_name"]
    with pytest.raises(ValueError, match="not a hierarchy proof"):
        runner._load_head_document(tmp_path, "h" * 64)


def test_the_hierarchy_chains_only_on_a_live_head_of_its_root(tmp_path):
    from gpuwm import hrrr_hierarchy_direct as hierarchy

    snapshots = _snapshots(3)
    # No head: a sealed or unchained root is built one-shot, as before.
    assert hierarchy.chained_root_head(tmp_path / "absent") is None
    writer = _tree_writer(tmp_path, snapshots, proof_name="proof.json",
                          name="root", stop_after=0)
    root = tmp_path / "root"
    assert hierarchy.chained_root_head(root)["head_sha256"] \
        == writer.head_sha256
    writer.fail(RuntimeError("stopped for the test"))
    # A root whose producer failed is never a head to build on.
    assert hierarchy.chained_root_head(root) is None


def test_the_one_shot_receipt_carries_the_root_seal_outside_its_provenance():
    """The root's sealed content digest is the one fact a tree's head lacks.

    It moves out of the receipt's provenance (which the head binds) into
    the seal-only ``root_preparation``, and d01's user metadata names it in
    a mapping the chained seal completes; the WRF export, written at the
    seal, keeps it in its input provenance.
    """

    from gpuwm import hrrr_hierarchy_direct as hierarchy

    assert hierarchy._root_preparation_record("c" * 64) == {
        "prepared_content_sha256": "c" * 64}
    assert hierarchy._root_preparation_metadata(None) == {
        "root_preparation": {}}
    assert hierarchy._root_preparation_metadata("c" * 64) == {
        "root_preparation": {"prepared_content_sha256": "c" * 64}}
    assert hierarchy._export_provenance({"a": 1}, "c" * 64) == {
        "a": 1, "root_prepared_content_sha256": "c" * 64}
    source = inspect.getsource(hierarchy.prepare_hrrr_hierarchy)
    assert '"root_prepared_content_sha256": restored' not in source
    assert '"root_preparation": _root_preparation_record(' in source
    tail = inspect.getsource(hierarchy._chained_hierarchy_tail)
    assert 'proof_name=RECEIPT_NAME' in tail
    assert 'seal_completes=("root_preparation",)' in tail
    # A posted tree relays each root marker before the root seals, so the
    # seal must also hold the exact markers the relay consumed.
    assert "verify_seal(c.root_preparation, head=c.root_head," in tail
    assert "consumed=root_stream.consumed_markers())" in tail


def test_the_tree_chain_starts_the_hierarchy_on_the_root_head(
        tmp_path, monkeypatch):
    """The hierarchy starts once the root preparation publishes this run's
    head; the tree forecast binds the tree's head; a stopped tree forecast
    stops the root preparation."""

    import threading
    import time as clock

    from gpuwm import runplan

    prep_root = tmp_path / "root"
    tree_root = tmp_path / "tree"
    order = []
    release_root = threading.Event()

    def preparation():
        order.append("root start")
        writer = _tree_writer(tmp_path, _snapshots(3), proof_name="proof.json",
                              name="root")
        release_root.wait(10)
        order.append("root sealed")
        _seal(writer)
        return {"root": "sealed"}

    def hierarchy():
        # Started on the root's head, before the root preparation seals.
        assert (prep_root / "boundary-stream" / "head.json").is_file()
        assert not (prep_root / "proof.json").exists()
        order.append("hierarchy start")
        writer = _tree_writer(tmp_path, _snapshots(3),
                              proof_name="receipt.json", name="tree")
        # The forecast is bound before the root seals.
        deadline = clock.monotonic() + 10
        while "forecast" not in order and clock.monotonic() < deadline:
            clock.sleep(0.05)
        release_root.set()
        _seal(writer)
        order.append("tree sealed")
        return tree_root

    forecasts = []
    monkeypatch.setattr(
        runplan, "_hrrr_tree_forecast",
        lambda **kwargs: (order.append("forecast"),
                          forecasts.append(kwargs))[1])
    observer = SimpleNamespace(finish_stage=lambda **k: None, events=None)
    prepared, chained = runplan._hrrr_tree_chain(
        prep_root=prep_root, tree_root=tree_root, preparation=preparation,
        hierarchy=hierarchy, config_path=tmp_path / "c.toml",
        forecast_dir=tmp_path / "run", observer=observer)
    assert order.index("hierarchy start") > order.index("root start")
    assert order.index("forecast") < order.index("root sealed")
    assert chained == read_head(tree_root)["head_sha256"]
    assert forecasts[0]["head_sha256"] == chained
    assert prepared["root"] == {"root": "sealed"}


def test_a_stopped_tree_forecast_stops_the_root_preparation(
        tmp_path, monkeypatch):
    from gpuwm import runplan

    prep_root = tmp_path / "root"
    tree_root = tmp_path / "tree"
    stopped = {}

    def preparation():
        writer = _tree_writer(tmp_path, _snapshots(3), proof_name="proof.json",
                              name="root", stop_after=0)
        import time as clock

        deadline = clock.monotonic() + 10
        while clock.monotonic() < deadline:
            try:
                writer.check_stop()
            except boundary_stream.BoundaryStreamStopped as error:
                stopped["root"] = str(error)
                writer.fail(error)
                raise
            clock.sleep(0.05)
        raise AssertionError("the root preparation was never stopped")

    def hierarchy():
        writer = _tree_writer(tmp_path, _snapshots(3),
                              proof_name="receipt.json", name="tree",
                              stop_after=0)
        import time as clock

        deadline = clock.monotonic() + 10
        while clock.monotonic() < deadline:
            try:
                writer.check_stop()
            except boundary_stream.BoundaryStreamStopped as error:
                writer.fail(error)
                raise RuntimeError("hierarchy stage exited 1") from error
            clock.sleep(0.05)
        raise AssertionError("the tree was never stopped")

    def forecast(**kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(runplan, "_hrrr_tree_forecast", forecast)
    observer = SimpleNamespace(finish_stage=lambda **k: None, events=None)
    with pytest.raises((KeyboardInterrupt, RuntimeError)):
        runplan._hrrr_tree_chain(
            prep_root=prep_root, tree_root=tree_root,
            preparation=preparation, hierarchy=hierarchy,
            config_path=tmp_path / "c.toml", forecast_dir=tmp_path / "run",
            observer=observer)
    assert "stopped" in stopped["root"]
