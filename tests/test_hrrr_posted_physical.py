"""HRRR posted capture holds actual lead evidence and precedes native real."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.ensemble.hrrr_physical_contract import hrrr_physical_field_contract
from gpuwm.ensemble.physical_store import physical_static_identity
from gpuwm.ensemble.posted_physical import PostedPhysicalStream, PhysicalFramePending
from gpuwm.ensemble.recipes import SourceTrajectory
from gpuwm.ingest.horiz import HorizontalSnapshot
from tools import hrrr_single_domain_benchmark as benchmark


def test_ordinary_source_cli_reaches_hrrr_posted_wrapper_without_final_manifest(tmp_path):
    from gpuwm import source_cli
    argv = ["--source", "hrrr", "--source-root", str(tmp_path / "raw"),
        "--as-posted", str(tmp_path / "raw/posting"),
        "--namelist-input", str(tmp_path / "namelist.input"),
        "--valid-time", "2024-05-21_12:00:00", "--output-root", str(tmp_path / "prepared"),
        "--static-cache", str(tmp_path / "static.npz"), "--static-receipt", str(tmp_path / "static.json")]
    args = source_cli._parser().parse_args(argv)
    assert source_cli._required_hrrr_args(args) == []
    assert source_cli.prepares_as_posted("hrrr")
    command = source_cli._hrrr_command(args)
    assert command[command.index("--as-posted") + 1] == str(tmp_path / "raw/posting")
    assert "--source-manifest" not in command and "None" not in command
    args = source_cli._parser().parse_args(argv + ["--source-manifest", str(tmp_path / "final.json"),
                                                "--source-manifest-sha256", "a" * 64])
    assert any("writes its source manifest at the seal" in error for error in source_cli._required_hrrr_args(args))


def _capture(tmp_path):
    from test_native_hrrr_posted import _marker, _decoded
    start = datetime(2026, 9, 30, 18)
    cycle = start - timedelta(hours=6)
    grid = {"mass_shape": [2, 3], "fixture": "hrrr-posted-native"}
    contract = hrrr_physical_field_contract(grid, evidence={
        "native_mapper": "a"*64, "ordinary_source_input_plan": "b"*64,
        "water_temperature_assembly": "c"*64, "native_decoder_executable": "d"*64})
    stream = PostedPhysicalStream.create(
        tmp_path/"physical", trajectory=SourceTrajectory("hrrr", cycle.replace(tzinfo=timezone.utc)),
        valid_times=[(start+timedelta(hours=i)).replace(tzinfo=timezone.utc) for i in range(3)],
        grid_identity=grid, source_identity={"input_manifest_sha256": "b"*64,
            "static_identity": physical_static_identity({"HGT_M": np.zeros((2,3))})},
        field_contract=contract, input_plan_sha256="b"*64)
    admitted = SimpleNamespace(markers={6: _marker(6)})
    decoded = {6: _decoded(6)}
    capture = benchmark._PostedHrrrCapture(stream, cycle=cycle, source_forecast_hours=(6,7,8),
                                          admitter=admitted, decoded_records=decoded)
    fields = {"TT": np.full((50,2,3), 280., "f4"),
              "PRES": np.broadcast_to(np.linspace(100000,10000,50,dtype="f4")[:,None,None], (50,2,3)).copy()}
    snapshot = HorizontalSnapshot(start, np.arange(1.,51.), fields)
    return capture, snapshot, admitted, decoded


def test_posted_capture_ready_before_future_or_seal_and_exact_before_mutation(tmp_path, monkeypatch):
    from gpuwm.ingest import real
    from gpuwm.core import grid as grid_module
    from test_native_hrrr_posted import _marker, _decoded
    capture, met, admitted, decoded = _capture(tmp_path)
    original = met.fields["TT"].tobytes()
    preprocess = SimpleNamespace(receipt=lambda: {"backend": "cpu", "workers": 1})
    monkeypatch.setattr(benchmark, "_map_snapshot", lambda *a, **k: (met, 0.))
    monkeypatch.setattr(grid_module, "make_vertical_coord", lambda *a, **k: object())
    def initialize(snapshot, *args, **kwargs):
        stored, receipt = capture.stream.require(snapshot.valid_time.replace(tzinfo=timezone.utc))
        assert stored.read(0).fields["TT"].tobytes() == original
        assert not capture.stream._marker(1).exists()
        assert not (capture.stream.root/"physical-seal.json").exists()
        snapshot.fields["TT"][:] += np.float32(7.)
        return SimpleNamespace(state=SimpleNamespace(set_map_coriolis=lambda *a, **k: None))
    monkeypatch.setattr(real, "initialize_real", initialize)
    static = {name: np.ones((2,3)) for name in
              ("HGT_M", "LANDMASK", "MAPFAC_M", "MAPFAC_U", "MAPFAC_V", "F", "E", "SINALPHA", "COSALPHA")}
    benchmark._initialize_state(SimpleNamespace(valid_time=met.valid_time),
        SimpleNamespace(run=SimpleNamespace(nz=2,hybrid_opt=2,etac=.2)), object(), static,
        np.asarray([1,.5,0]), {}, p_top=5000, preprocess_backend=preprocess,
        state_backend="preprocess", physical_output=capture)
    stored, receipt = capture.stream.require(met.valid_time.replace(tzinfo=timezone.utc))
    assert stored.read(0).fields["TT"].tobytes() == original
    assert receipt["marker"]["posted_leads"] and receipt["marker"]["decoded_leads"]
    with pytest.raises(PhysicalFramePending):
        capture.stream.require((met.valid_time+timedelta(hours=1)).replace(tzinfo=timezone.utc))
    from dataclasses import replace
    for lead in (7,8):
        admitted.markers[lead], decoded[lead] = _marker(lead), _decoded(lead)
        capture.write(replace(met, valid_time=met.valid_time+timedelta(hours=lead-6)))
    capture.seal()
    assert (capture.stream.root/"physical-seal.json").exists()


def test_capture_refuses_missing_decoded_authority_before_publishing(tmp_path):
    capture, met, admitted, decoded = _capture(tmp_path)
    decoded.clear()
    with pytest.raises(ValueError, match="posted and decoded"):
        capture.write(met)
    assert not capture.stream._marker(0).exists()
    assert not (capture.stream.root/"frames").exists()


def test_posted_output_keeps_the_ordinary_chained_route(monkeypatch):
    from tools.prepare_hrrr_wrf import _chains
    from gpuwm.ingest import boundary_stream
    monkeypatch.setattr(boundary_stream, "chained_enabled", lambda: True)
    monkeypatch.setattr(boundary_stream, "forecast_installed", lambda: True)
    args = SimpleNamespace(sealed_prepared_cache=False, physical_input_store=None,
                           physical_output_store="capture", as_posted="posting")
    assert _chains(args)
    args.as_posted = None
    assert not _chains(args)


def test_posted_native_door_refuses_older_decoder_before_launch(monkeypatch, tmp_path):
    from gpuwm import bridges
    from tools.prepare_hrrr_wrf import _decoder
    path = tmp_path / "hrrr_grib2_bridge"
    path.write_bytes(bridges.BRIDGE_ABI_MARKERS["hrrr_grib2_bridge"])
    monkeypatch.setattr(bridges, "find_bridge", lambda name: path)
    assert bridges.resolve_source_decoder("hrrr") == path
    with pytest.raises(bridges.DecoderContractError, match="--series-workers-posted"):
        _decoder({}, as_posted=True)
    with path.open("ab") as stream:
        stream.write(b"--series-workers-posted WORKERS SERIES_TSV OUTPUT_DIR SIGNAL_DIR ADMIT_DIR")
    assert _decoder({}, as_posted=True) == path


def test_early_input_plan_keeps_the_observed_native_route_at_head_publication():
    import hashlib
    from copy import deepcopy
    from gpuwm.hrrr_prepared_bundle import _canonical
    from gpuwm.ingest.boundary_stream import input_plan, input_plan_sha256
    from test_native_hrrr_posted import _plan_manifest
    manifest = _plan_manifest()
    plan = input_plan(manifest, lead_role_prefix="hrrr-f", route_table_sha256="7"*64,
                      derived_roles=("bridge", "source_manifest"))
    plan_before = deepcopy(plan)
    bundle = {"manifest": manifest, "proof_head": {"preprocessing": {"vertical_interpolation": []}}}
    observed = {"backend": "cpu", "vertical_interpolation": [
        {"backend": "cpu", "source_levels": 50, "column_levels": 51}]}
    benchmark._complete_posted_head_receipts(bundle, metadata={}, preprocess_receipt=observed)
    assert bundle["proof_head"]["preprocessing"] == observed
    assert bundle["proof_head"]["preprocessing_receipt_sha256"] == hashlib.sha256(
        _canonical(observed).encode()).hexdigest()
    assert plan == plan_before
    assert input_plan_sha256(plan) == input_plan_sha256(plan_before)


def test_shared_source_requires_original_window_and_waits_only_for_needed_interval():
    from tools.hrrr_posted_reuse import SharedPostedHrrr
    cycle = datetime(2026, 9, 30, 12)
    times = tuple((cycle + timedelta(hours=hour)).replace(tzinfo=timezone.utc)
                  for hour in (6, 7, 8))
    waited, consumed = [], []
    context = SimpleNamespace(
        trajectory=SourceTrajectory("hrrr", cycle.replace(tzinfo=timezone.utc)),
        prepared_head={"basis": {"as_posted": {"forcing_leads": [6, 7, 8]}}},
        physical_stream=SimpleNamespace(times=times, require=consumed.append),
        require_interval=waited.append)
    context.verify = lambda: context
    with pytest.raises(ValueError, match="requested source window"):
        SharedPostedHrrr(context, cycle=cycle, source_forecast_hours=(6, 7))
    with pytest.raises(ValueError, match="requested source window"):
        SharedPostedHrrr(context, cycle=cycle + timedelta(hours=1), source_forecast_hours=(6, 7, 8))
    shared = SharedPostedHrrr(context, cycle=cycle, source_forecast_hours=(6, 7, 8))
    assert shared.acquire(0).valid_time == times[0].replace(tzinfo=None)
    assert waited == [] and consumed == [times[0]]
    assert shared.acquire(2).valid_time == times[2].replace(tzinfo=None)
    assert waited == [1] and consumed == [times[0], times[2]]


def test_shared_source_worker_receipt_needs_explicit_reuse_and_zero_decoders():
    from copy import deepcopy
    from tools.prepare_hrrr_wrf import _validated_worker_receipts
    from test_prepare_hrrr_wrf import _cuda_preparation_receipt
    report = {"status": "PASS", "preparation": _cuda_preparation_receipt(),
              "pipeline": {"workers": {"requested": "2", "selected": 0,
                                       "operation": "reused_posted_native_source"}}}
    options = dict(selected_backend="cuda", requested_preprocess_workers=None,
                   requested_pipeline_workers="2", final_hour=12)
    with pytest.raises(RuntimeError, match="decoder worker"):
        _validated_worker_receipts(report, **options)
    assert _validated_worker_receipts(report, reused_posted_source=True, **options)[2]["selected"] == 0
    for key, value in (("selected", 2), ("operation", "reused_sealed_native_bridge")):
        changed = deepcopy(report)
        changed["pipeline"]["workers"][key] = value
        with pytest.raises(RuntimeError, match="native source reuse"):
            _validated_worker_receipts(changed, reused_posted_source=True, **options)
