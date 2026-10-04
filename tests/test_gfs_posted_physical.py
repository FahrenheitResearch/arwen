"""The GFS native physical seam consumes posted members before future seals."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta, timezone
import hashlib
import json
from pathlib import Path
from threading import Event, Lock, current_thread, main_thread
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm import gfs_direct
from gpuwm.ensemble.posted_physical import PostedPhysicalProvider, PostedPhysicalStream
from gpuwm.ensemble.recipes import RecipeMember, SourceRecipe
from gpuwm.ingest import boundary_stream, prepared_cache
from test_gfs_physical_store import physical_route
from test_posted_preparation import _ReplayedFetch


@pytest.fixture
def posted_route(physical_route, monkeypatch, tmp_path):
    _prepare, snapshots, arguments, initialized = physical_route
    replay = _ReplayedFetch(tmp_path/"posted-fetch", arguments["experiment_config"], (0, 1, 2, 3))
    replay.publish(0)
    original_run = gfs_direct.subprocess.run
    def decode(command, **kwargs):
        if Path(command[0]) != arguments["bridge"]:
            return original_run(command, **kwargs)
        output = Path(command[2] if command[1] == "--merge-batches" else command[3])
        output.mkdir()
        for name in ("gate.tsv", "inventory.tsv", "decoded-sha256.tsv"):
            (output/name).write_text("posted physical native seam fixture\n")
        return SimpleNamespace(returncode=0, stdout="PASS source fixture", stderr="")
    monkeypatch.setattr(gfs_direct.subprocess, "run", decode)
    def frame(hour):
        first = snapshots[0]
        return replace(first, valid_time=first.valid_time+timedelta(hours=hour),
                       fields={**first.fields, "TT": first.fields["TT"]+np.float32(hour),
                               "RH": first.fields["RH"]+np.float32(hour)})
    monkeypatch.setattr(gfs_direct, "_load_bridge_snapshots",
                        lambda _root, _cycle, batch, *_a, **_k: tuple(frame(hour) for hour, _ in batch))
    # This fixture controls native array construction; keep its checked
    # source view explicit while the real archived proof exercises full
    # forecast preflight over actual native artifacts.
    from gpuwm.ensemble import posted_native
    from gpuwm.experiment import load_experiment
    original_static = gfs_direct._static_from_geog
    original_grid = gfs_direct._validate_grid_and_vertical_contract
    def checked(context, **kwargs):
        context.verify()
        exp = load_experiment(arguments["experiment_config"])
        grid = original_grid(exp, arguments["wps_namelist"])
        static, _, _ = original_static()
        base = frame(0)
        reader = SimpleNamespace(header={"metadata": {
            "met_fields": list(base.fields), "surface_fields": ["TSK"], "user": {}}},
            read_array=lambda name: base.fields[name[4:]].copy() if name.startswith("met/")
                else base.fields["SKINTEMP"].copy())
        return SimpleNamespace(experiment=exp, grid=grid, static=static,
            cache_reader=reader, proof=deepcopy(context.prepared_head["basis"]["proof_head"]),
            static_path=context.prepared_root/"native-static.npz",
            geometry_receipt_path=context.prepared_root/"geometry-receipt.json", geometry_receipt={},
            cache_identity=context.prepared_head["basis"]["cache"]["identity"],
            boundary_interval_seconds=3600)
    monkeypatch.setattr(posted_native, "checked_source_inputs", checked)
    monkeypatch.setattr(boundary_stream, "_host_available", lambda: 64*1024**3)
    monkeypatch.setenv("GPUWM_CHAINED_PREP", "1")
    original_head = boundary_stream.PreparedTreeWriter.write_head
    captured_heads = {}
    on_head = [None]
    def head(writer, **kwargs):
        result = original_head(writer, **kwargs)
        captured_heads[writer.output_root.name] = {
            "fetch_leads": [value[0] for value in replay.published],
            "physical_ready": sorted(path.name for path in (tmp_path/"physical"/"ready").glob("*.json")),
            "head": deepcopy(writer.head)}
        if on_head[0] is not None:
            on_head[0](writer.output_root.name)
        return result
    monkeypatch.setattr(boundary_stream.PreparedTreeWriter, "write_head", head)

    class Cache:
        """Cheap arrays; real source head/segment/seal verification stays active."""
        def __init__(self, directory, *, identity, **_kwargs):
            self.directory, self.identity = Path(directory), identity
            self.metadata = None
        def move(self, directory):
            self.directory = Path(directory)
        def write_head(self, **kwargs):
            self.directory.mkdir(parents=True)
            self.metadata = {"user": deepcopy(kwargs.get("metadata") or {})}
            result = {"identity": deepcopy(self.identity), "metadata": deepcopy(self.metadata),
                      "arrays": {}, "payload_bytes": 0, "lbc": kwargs["lbc"],
                      "setup_core_fingerprint": "0"*64}
            if kwargs.get("seal_completes"):
                result[prepared_cache.SEAL_COMPLETES_KEY] = list(kwargs["seal_completes"])
            return result
        def write_segment(self, index, interval):
            return {"index": index, "start_seconds": float(interval.start_seconds),
                    "end_seconds": float(interval.end_seconds), "fields": sorted(interval.fields),
                    "arrays": {}, "payload_bytes": 0, "prefix": {}}
        def seal(self, *, identity=None, completed_metadata=None, posted_user_metadata=None):
            if identity is not None:
                self.identity = identity
            metadata = deepcopy(self.metadata)
            metadata["user"].update(completed_metadata or {})
            metadata["user"].update(posted_user_metadata or {})
            basis = json.loads(json.dumps({"schema": prepared_cache.PREPARED_CACHE_SCHEMA,
                "identity": self.identity, "metadata": metadata, "arrays": {}, "payload_bytes": 0}, default=str))
            digest = hashlib.sha256(prepared_cache._canonical(basis).encode()).hexdigest()
            (self.directory/"header.json").write_text(json.dumps({**basis, "content_sha256": digest}))
            return {"schema": "gpuwm-prepared-cache-v1", "status": "BUILT", "content_sha256": digest,
                    "array_count": 0, "payload_bytes": 0}
    monkeypatch.setattr(prepared_cache, "PreparedCacheStream", Cache)
    allow_future, source_head = Event(), Event()
    publish_lock = Lock()
    def publish_future():
        with publish_lock:
            present = {row[0] for row in replay.published}
            for hour in (1, 2, 3):
                if hour not in present:
                    replay.publish(hour)
    class Leads(boundary_stream.PostedLeads):
        def __init__(self, *args, **kwargs):
            def sleep(_seconds):
                if not allow_future.wait(20):
                    raise RuntimeError("test member never published its initial head")
                publish_future()
            super().__init__(*args, **{**kwargs, "sleep": sleep})
    monkeypatch.setattr(boundary_stream, "PostedLeads", Leads)
    def prepare(name, **options):
        return gfs_direct.prepare_gfs_wrf(**{**arguments,
            "series": replay.series, "input_manifest": tmp_path/"posted-input-manifest.json",
            "input_manifest_sha256": None, "output_root": tmp_path/name,
            "as_posted": replay.posting, **options}, stock_wrf_export=False)
    return SimpleNamespace(prepare=prepare, replay=replay, frame=frame, initialized=initialized,
        heads=captured_heads, on_head=on_head, allow_future=allow_future, source_head=source_head,
        publish_future=publish_future, root=tmp_path, arguments=arguments)


def test_gfs_posted_capture_is_ready_before_future_source_leads(posted_route):
    route = posted_route
    def head(name):
        assert name == "source"
        stream = PostedPhysicalStream(route.root/"physical")
        first, _ = stream.require(stream.times[0])
        assert first.read(0).fields["RH"].tobytes() == route.frame(0).fields["RH"].tobytes()
        assert not stream._marker(1).exists()
        assert not (route.root/"source"/"proof.json").exists()
        route.allow_future.set()
    route.on_head[0] = head
    route.prepare("source", physical_output_store=route.root/"physical")
    assert route.heads["source"]["fetch_leads"] == [0]
    assert route.heads["source"]["physical_ready"] == ["00000.json"]
    stream = PostedPhysicalStream(route.root/"physical")
    for hour in range(4):
        actual = stream.require(stream.times[hour])[0].read(0)
        assert actual.specific_humidity_authority is False
        assert all(value.tobytes() == route.frame(hour).fields[name].tobytes() for name, value in actual.fields.items())
    boundary_stream.verify_seal(route.root/"source", head=boundary_stream.read_head(route.root/"source"))


@pytest.mark.parametrize("use_factory", [False, True])
def test_gfs_member_head_and_boundaries_consume_incremental_physical_provider(posted_route, monkeypatch, use_factory):
    route = posted_route
    def head(name):
        if name == "source":
            route.source_head.set()
        elif name == "member":
            assert not (route.root/"source"/"proof.json").exists()
            assert route.replay.published[0][0] == 0 and len(route.replay.published) == 1
            route.allow_future.set()
    route.on_head[0] = head
    with ThreadPoolExecutor(max_workers=1) as executor:
        source_job = executor.submit(route.prepare, "source", physical_output_store=route.root/"physical")
        try:
            assert route.source_head.wait(20), "source did not publish its initial head"
            stream = PostedPhysicalStream(route.root/"physical")
            native_calls = []
            # The source already owns these stages. Any repeat by the member
            # is a concrete regression, even if it returns equal fields.
            original_native = gfs_direct.initialize_real
            def initialize(*args, **kwargs):
                if current_thread() is main_thread():
                    native_calls.append(args[0].valid_time)
                return original_native(*args, **kwargs)
            monkeypatch.setattr(gfs_direct, "initialize_real", initialize)
            def guard(function, name):
                def invoke(*args, **kwargs):
                    if current_thread() is main_thread():
                        raise AssertionError("member repeats shared preparation: "+name)
                    return function(*args, **kwargs)
                return invoke
            for name in ("_PostedGfsSeries", "_load_bridge_snapshots", "_seal_as_posted_inputs",
                         "_static_from_geog", "_load_static", "_survey_static_catalog",
                         "preprocess_land_surface_soil", "interpolate_era5_to_lambert"):
                monkeypatch.setattr(gfs_direct, name, guard(getattr(gfs_direct, name), name))
            start, end = stream.times[0], stream.times[-1]
            recipe = SourceRecipe("control", stream.trajectory, start, end,
                                  (RecipeMember(23, 7023, stream.trajectory),))
            if use_factory:
                from gpuwm import source_cli
                from gpuwm.ensemble import source_preparation
                from gpuwm.fetch_routes import PREP_ARGUMENTS_SCHEMA
                (route.replay.out/"prep-arguments.json").write_text(json.dumps({
                    "schema": PREP_ARGUMENTS_SCHEMA, "source": stream.trajectory.source,
                    "cycle": stream.trajectory.cycle.isoformat(), "member": None,
                    "as_posted": True, "posting": str(route.replay.posting),
                    "argv": ["--source", stream.trajectory.source, "--gfs-series", str(route.replay.series),
                             "--cycle", route.arguments["cycle"]]}))
                specification = source_preparation.PostedSourcePreparation.from_acquisition(
                    stream.trajectory, acquisition_root=route.replay.out,
                    prepared_root=route.root/"source", physical_root=route.root/"physical",
                    native_arguments=("--bridge", str(route.arguments["bridge"]),
                        "--wps-namelist", str(route.arguments["wps_namelist"]),
                        "--experiment-config", str(route.arguments["experiment_config"]),
                        "--geog-root", str(route.arguments["geog_root"]),
                        "--preprocess-backend", "cpu", "--no-stock-wrf-export"))
                factory = source_preparation.PostedPreparationFactory.publish(
                    recipe, {stream.trajectory.identity: specification}, root=route.root/"provider")
                factory = source_preparation.PostedPreparationFactory.open(factory.provider.root)
                ordinary_subprocess = source_preparation.subprocess.run
                commands = []
                def subprocess(command, **kwargs):
                    if list(command[1:3]) == ["-m", "gpuwm.source_cli"]:
                        commands.append(list(command))
                        # Keep the actual source CLI parser/adapter command and
                        # native main; only the child-process boundary is in
                        # process so controlled decode/array fixtures remain.
                        parsed = source_cli._parser().parse_args(command[3:])
                        native = source_cli._gfs_command(parsed)
                        code = gfs_direct.main(native[3:])
                        if code:
                            raise RuntimeError(f"native factory member returned {code}")
                        return SimpleNamespace(returncode=code)
                    return ordinary_subprocess(command, **kwargs)
                monkeypatch.setattr(source_preparation.subprocess, "run", subprocess)
                result = factory.prepare_member(23, output_root=route.root/"member")
                assert result["member_index"] == 23
                assert len(commands) == 1
            else:
                provider = PostedPhysicalProvider(recipe, {stream.trajectory.identity: stream})
                provider.write_plan(route.root/"provider", prepared_roots={stream.trajectory.identity: route.root/"source"})
                route.prepare("member", physical_input_provider=provider.root, physical_member_index=23)
            assert native_calls == [route.frame(hour).valid_time for hour in range(4)]
        finally:
            route.allow_future.set()
            route.publish_future()
        source_job.result(timeout=20)
    assert route.heads["member"]["fetch_leads"] == [0]
    head_document = boundary_stream.read_head(route.root/"member")
    assert head_document["basis"]["ensemble_physical"]["member_index"] == 23
    verified = boundary_stream.verify_seal(route.root/"member", head=head_document)
    assert verified
    header = json.loads((route.root/"member"/"prepared-cache"/"header.json").read_text())
    binding = header["identity"]["source_identity"]["ensemble_posted_physical_input"]
    assert binding["provider_receipt"]["member_seed"] == 7023
    assert "as-posted:" not in json.dumps(binding)
    for index in range(3):
        segment = json.loads(boundary_stream.segment_marker_path(route.root/"member", index).read_text())
        assert "ensemble_physical" in segment


def test_gfs_provider_requires_original_member_index_and_posted_mode(physical_route, tmp_path):
    _prepare, _snapshots, arguments, initialized = physical_route
    with pytest.raises(ValueError, match="original member index"):
        gfs_direct.prepare_gfs_wrf(**arguments, physical_input_provider=tmp_path/"provider")
    assert initialized == []


@pytest.mark.parametrize("decoder", ["omitted", "missing", "different_bytes", "different_name"])
def test_gfs_member_reuses_pinned_decoder_without_installation(posted_route, decoder):
    route = posted_route
    route.allow_future.set()
    route.prepare("source", physical_output_store=route.root/"physical")
    stream = PostedPhysicalStream(route.root/"physical")
    recipe = SourceRecipe("control", stream.trajectory, stream.times[0], stream.times[-1],
                          (RecipeMember(23, 7023, stream.trajectory),))
    provider = PostedPhysicalProvider(recipe, {stream.trajectory.identity: stream})
    provider.write_plan(route.root/"provider", prepared_roots={stream.trajectory.identity: route.root/"source"})
    bridge = None
    if decoder != "omitted":
        folder = route.root/"consumer-tools"
        folder.mkdir()
        bridge = folder/route.arguments["bridge"].name
        if decoder == "different_bytes":
            bridge.write_bytes(b"another decoder")
        elif decoder == "different_name":
            bridge = folder/"another-decoder"
    kwargs = dict(bridge=bridge, physical_input_provider=provider.root, physical_member_index=23)
    if decoder.startswith("different"):
        with pytest.raises(ValueError, match="decoder differs from its pinned"):
            route.prepare("member", **kwargs)
        assert not (route.root/"member").exists()
    else:
        route.prepare("member", **kwargs)
        head = boundary_stream.read_head(route.root/"member")
        boundary_stream.verify_seal(route.root/"member", head=head)
        assert head["basis"]["as_posted"]["input_plan"] == provider.source_context(23).source_plan
