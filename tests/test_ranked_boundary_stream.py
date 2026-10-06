"""Ranked forcing stays lazy and fails before any rank crosses a seam."""
from types import SimpleNamespace

import pytest

from gpuwm.ingest import boundary_stream
from tilestream.ranks import RankedRun


def test_preparation_telemetry_can_complete_at_the_seal(monkeypatch, tmp_path):
    from test_boundary_stream import _chained_tree, _snapshots

    publish = boundary_stream.PreparedTreeWriter.publish
    telemetry = {"preparation_parallelism": {"effective_workers": 8},
                 "forcing_stage_timings": [{"forcing_index": 0, "total_seconds": 1.25}]}

    def completed(writer, proof):
        return publish(writer, {**proof, **telemetry})

    monkeypatch.setattr(boundary_stream.PreparedTreeWriter, "publish", completed)
    writer, output = _chained_tree(tmp_path, _snapshots(3))
    head = boundary_stream.read_head(output)
    assert not set(telemetry) & set(head["basis"]["proof_head"])
    boundary_stream.verify_seal(output, head=head)


@pytest.mark.parametrize("kind", ["single", "tree"])
@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_ranked_head_prices_the_producer_card_and_host_store(monkeypatch, kind, backend):
    from gpuwm.core import devices_memory, preflight

    gib = 1 << 30
    prices = []
    exp = SimpleNamespace(
        devices=SimpleNamespace(enabled=True, device_ids=lambda: (2, 3)),
        domains=(object(),) * (1 if kind == "single" else 2))

    def price(experiment, **kwargs):
        prices.append((experiment, kwargs))
        return {"cards": [{"card": 2, "total_bytes": 7 * gib},
                          {"card": 3, "total_bytes": 20 * gib}],
                "host_store_bytes": 9 * gib, "host_staging_bytes": gib}

    monkeypatch.setattr(devices_memory, "estimate_devices", price)
    monkeypatch.setattr(devices_memory, "estimate_devices_tree", price)
    monkeypatch.setattr(preflight, "admission_estimate", lambda *_a, **_k:
                        pytest.fail("ranked preparation priced a whole resident domain"))
    decision = boundary_stream.chained_admission(
        experiment=exp, backend=backend, source="mapped", device_bytes=4 * gib,
        card=(8 * gib, 4 * gib))
    assert decision["admitted"]
    assert decision["forecast_store_host_bytes"] == 10 * gib
    assert prices == [(exp, {"streaming_boundaries": True, "source": "mapped"})]
    if backend == "cuda":
        assert decision["forecast_bytes"] == 7 * gib
        assert decision["producer_card"] == 2
        refused = boundary_stream.chained_admission(
            experiment=exp, backend=backend, device_bytes=5 * gib,
            card=(8 * gib, 4 * gib))
        assert not refused["admitted"]


def test_chunked_producer_reserves_next_hour_peak(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(
        get_default_memory_pool=lambda: SimpleNamespace(total_bytes=lambda: 7)))
    assert boundary_stream.producer_device_bytes("cpu") is None
    assert boundary_stream.producer_device_bytes("cuda") == 7
    assert boundary_stream.producer_device_bytes("cuda", selection={
        "chunking": {"rows": 2}, "device_fit": {"need_bytes": 15}}) == 15
    assert boundary_stream.producer_device_bytes("cuda", selection={
        "chunking": {"rows": 2}, "device_fit": {"need_bytes": 3}}) == 7


def test_bounded_head_allows_cards_selected_by_the_forecast(monkeypatch, tmp_path):
    import sys
    from gpuwm.core import preflight
    from test_boundary_stream import _initial, _met, _snapshots, _frames

    gib = 1 << 30
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(
        get_default_memory_pool=lambda: SimpleNamespace(total_bytes=lambda: 7),
        cuda=SimpleNamespace(Device=lambda: SimpleNamespace(id=0))))
    monkeypatch.setattr(preflight, "admission_estimate", lambda *_a, **_k:
                        SimpleNamespace(peak_envelope_bytes=100 * gib))
    monkeypatch.setattr(boundary_stream, "_host_available", lambda: 128 * gib)
    monkeypatch.setattr(boundary_stream, "process_memory_bytes", lambda: (gib, gib))
    identity = {"uuid": "a" * 32, "pci_bus_id": "00000000:42:00.0"}
    monkeypatch.setattr("gpuwm.core.device_probe.cuda_device_identity", lambda _dev: identity)
    staging = tmp_path / "staging"
    staging.mkdir()
    writer = boundary_stream.PreparedTreeWriter(
        staging=staging, output_root=tmp_path / "prepared", identity={}, chained=True)
    decision = writer.admit(
        experiment=SimpleNamespace(devices=SimpleNamespace(enabled=False)),
        backend="cuda", device_bytes=gib, card=(31 * gib, 0),
        preprocess_selection={"chunking": {"rows": 2}, "device_fit": {"need_bytes": gib},
                              "host_fit": {"producer_peak_bytes": 3 * gib}})
    assert decision["admitted"] and writer.chained
    assert not decision["configured_forecast_admitted"]
    assert decision["consumer_admission_required"]
    assert decision["producer_reserve_bytes"] == gib
    assert decision["producer_device_identity"] == identity
    frames = _frames(_snapshots(1))
    writer.write_head(initial_result=_initial(), met=_met(), lbc={
        "spec_bdy_width": 5, "spec_zone": 1, "relax_zone": 4,
        "schedule": [[0., 3600.], [3600., 7200.]], "fields": frames.inventory},
        proof_head={"schema": "test"}, forcing=frames)
    head = boundary_stream.read_head(writer.root)
    assert head["decision"]["consumer_admission_required"]
    assert head["decision"]["host"]["producer_host_bytes"] == 2 * gib
    assert not (writer.root / "proof.json").exists()
    writer.fail(RuntimeError("test finished"))


def test_ranked_forecast_leaves_released_producer_batch_free():
    from gpuwm.prepared_single_domain_forecast import _devices_stream_budgets

    free = {2: 20, 3: 20}
    assert _devices_stream_budgets(free, None) is free
    assert _devices_stream_budgets(free, {"decision": {"producer_bytes": 8}}) is free
    assert _devices_stream_budgets(free, {"decision": {
        "producer_reserve_bytes": 8, "producer_card": 2}}) == {2: 12, 3: 12}
    assert free == {2: 20, 3: 20}


@pytest.mark.parametrize("priced_peak,additional", [(40, 5), (50, 15), (30, 5)])
def test_host_reserves_future_peak_without_counting_current_rss_twice(
        monkeypatch, priced_peak, additional):
    monkeypatch.setattr(boundary_stream, "process_memory_bytes", lambda: (35, 40))
    result = boundary_stream.host_admission(
        forecast_bytes=30, available_bytes=50, producer_peak_bytes=priced_peak,
        producer_peak_required=True)
    assert result["producer_host_bytes"] == additional
    assert result["producer_resident_bytes"] == 35
    assert result["producer_observed_peak_bytes"] == 40
    assert result["producer_priced_peak_bytes"] == priced_peak
    assert result["admitted"]
    assert not boundary_stream.host_admission(
        forecast_bytes=41, available_bytes=50, producer_peak_bytes=priced_peak,
        producer_peak_required=True)["admitted"]


def test_bounded_head_refuses_an_unmeasured_future_host_peak(monkeypatch):
    monkeypatch.setattr(boundary_stream, "process_memory_bytes", lambda: (35, 40))
    assert not boundary_stream.host_admission(
        forecast_bytes=1, available_bytes=100, producer_peak_required=True)["admitted"]


def test_selected_store_keeps_the_producer_host_reservation(monkeypatch):
    from gpuwm.prepared_single_domain_forecast import (
        _stream_host_budget, _stream_reserved_machine, _stream_reserved_options,
    )
    from gpuwm.core.streaming import StreamingOptions
    from tilestream.autoplan import Machine

    monkeypatch.setattr("gpuwm.core.preflight.host_available_bytes", lambda: 70)
    head = {"decision": {"host": {"producer_host_bytes": 15}}}
    assert _stream_host_budget(70, head) == 55
    assert _stream_host_budget(None, head) is None
    machine = Machine(vram_bytes=20, host_bytes=100)
    reserved = _stream_reserved_machine(machine, head)
    assert reserved.vram_bytes == 20 and reserved.host_bytes == 55
    assert machine.host_bytes == 100
    options = StreamingOptions(mode="auto", host_budget_bytes=80)
    assert _stream_reserved_options(options, reserved, head).host_budget_bytes == 55


def test_late_native_decode_reserves_more_than_the_parent_real_peak(monkeypatch):
    gib = 1 << 30
    monkeypatch.setattr(boundary_stream, "process_memory_bytes", lambda: (3 * gib, 4 * gib))
    options = dict(forecast_bytes=4 * gib, producer_peak_bytes=5 * gib,
                   producer_peak_required=True)
    assert boundary_stream.host_admission(**options, available_bytes=24 * gib)["admitted"]
    refused = boundary_stream.host_admission(
        **options, available_bytes=24 * gib, producer_decode_bytes=20 * gib)
    assert not refused["admitted"]
    assert refused["producer_host_bytes"] == 22 * gib
    roomy = boundary_stream.host_admission(
        **options, available_bytes=100 * gib, producer_decode_bytes=20 * gib)
    assert roomy["admitted"] and roomy["producer_host_bytes"] == 22 * gib
    unpriced = boundary_stream.host_admission(
        **options, available_bytes=100 * gib, producer_decode_bytes=None)
    assert not unpriced["admitted"] and "future source decoder" in unpriced["reason"]


def test_native_decode_price_includes_components_and_one_process_baseline():
    import json
    from gpuwm.mapped_engine_bridge import _decode_memory_receipt

    mib = 1 << 20
    records = [{"memory_priced": True, "per_time_bytes": 200 * mib, "whole_per_time_bytes": 400 * mib,
                "process_rss_bytes": 80 * mib},
               {"memory_priced": True, "per_time_bytes": 300 * mib, "whole_per_time_bytes": 300 * mib,
                "process_rss_bytes": 400 * mib}]
    stderr = "\n".join("GPUWM_PREP_THREADS " + json.dumps(row) for row in records)
    receipt = _decode_memory_receipt(stderr, budget_supported=True)
    assert receipt["per_time_bytes"] == 500 * mib
    assert receipt["process_baseline_bytes"] == 80 * mib
    assert receipt["whole_per_time_bytes"] == 700 * mib
    assert receipt["one_time_peak_bytes"] == 1130 * mib
    assert receipt["budget_supported"]
    records[0]["per_time_bytes"] = 0
    assert _decode_memory_receipt(
        "GPUWM_PREP_THREADS " + json.dumps(records[0]), budget_supported=True) is None
    assert _decode_memory_receipt(stderr + "\nGPUWM_PREP_THREADS " + json.dumps(
        {"memory_priced": False, "per_time_bytes": 0, "whole_per_time_bytes": 0,
         "process_rss_bytes": 2 * mib}), budget_supported=True) is None
    assert _decode_memory_receipt(stderr + "\nGPUWM_PREP_THREADS " + json.dumps(
        {**records[1], "memory_priced": False}), budget_supported=True) is None
    small = _decode_memory_receipt("GPUWM_PREP_THREADS " + json.dumps(
        {"memory_priced": True, "per_time_bytes": 1, "whole_per_time_bytes": 3, "process_rss_bytes": 1}),
        budget_supported=True)
    assert small["one_time_peak_bytes"] == 64 * mib + 5


def test_posted_source_declines_unpriced_future_decode_and_omits_a_finished_one():
    from gpuwm.mapped_direct import _PostedMappedSource

    source = object.__new__(_PostedMappedSource)
    source._next, source.leads = 1, (0, 1, 2)
    source.batches = [SimpleNamespace(native_decode_memory={
        "one_time_peak_bytes": 200, "budget_supported": True})]
    assert source.future_decode_host_bytes() == 200
    source.batches[0].native_decode_memory["budget_supported"] = False
    assert source.future_decode_host_bytes() is None
    source._next = len(source.leads)
    assert source.future_decode_host_bytes() == 0


def test_native_decode_budget_is_per_call_and_requires_matching_capability(monkeypatch, tmp_path):
    import json
    import os
    import subprocess
    from gpuwm import mapped_engine_bridge as bridge

    observed = []
    supports = True
    def execute(command, **kwargs):
        if command[-1] == "capabilities":
            return subprocess.CompletedProcess(command, 0, json.dumps({
                "schema": bridge.CAPABILITIES_SCHEMA,
                "features": {"host_memory_budget": bridge.HOST_MEMORY_BUDGET_SCHEMA}
                            if supports else {}}), "")
        observed.append(kwargs.get("env"))
        return subprocess.CompletedProcess(command, 0, "", "")
    monkeypatch.setattr(bridge.subprocess, "run", execute)
    monkeypatch.delenv(bridge.ENGINE_MEMORY_BUDGET_ENV, raising=False)
    options = dict(mapping=tmp_path / "mapping.json", files=[], output=tmp_path / "decoded",
                   engine=tmp_path / "engine", memory_budget_bytes=123456)
    bridge.run_engine("compose", **options)
    assert observed[0][bridge.ENGINE_MEMORY_BUDGET_ENV] == "123456"
    assert bridge.ENGINE_MEMORY_BUDGET_ENV not in os.environ
    supports = False
    with pytest.raises(bridge.EngineUnavailable, match="cannot enforce"):
        bridge.run_engine("compose", **options)
    assert len(observed) == 1


def test_initial_frameset_fallback_takes_the_later_head_budget(monkeypatch, tmp_path):
    from gpuwm import mapped_engine_bridge
    from gpuwm.mapped_direct import _PostedMappedSource
    from test_mapped_composition import _engine_compose_harness

    harness = _engine_compose_harness(tmp_path, monkeypatch)
    original_open = mapped_engine_bridge.open_frameset
    fallbacks, calls = [], []
    def open_frames(directory, *, full_fallback=None, **kwargs):
        if full_fallback is not None:
            fallbacks.append(full_fallback)
        return original_open(directory, **kwargs)
    def execute(*_args, **kwargs):
        calls.append(kwargs.get("memory_budget_bytes"))
        return {"decode_memory": {"budget_supported": True, "one_time_peak_bytes": 500}}
    monkeypatch.setattr(mapped_engine_bridge, "open_frameset", open_frames)
    monkeypatch.setattr(mapped_engine_bridge, "run_engine", execute)
    bundle = harness.compose(tmp_path / "prepared", atmospheric_grids=(object(),))
    assert calls == [None] and bundle.future_decode_host_bytes() == 500
    source = object.__new__(_PostedMappedSource)
    source._next, source.leads, source.batches = 2, (0, 1), [bundle]
    # All leads are decoded, but their reader can still need whole donors.
    assert source.future_decode_host_bytes() == 500
    source.limit_future_decode(500)
    fallbacks[0]()
    assert calls == [None, 500]
    assert source.future_decode_host_bytes() == 0


@pytest.mark.parametrize("producer,identities,expected", [
    ({"uuid": "a"}, {0: {"uuid": "b"}, 1: {"uuid": "a"}}, {0: 20, 1: 12}),
    ({"uuid": "a"}, {0: {"uuid": "b"}, 1: {"uuid": "c"}}, {0: 20, 1: 20}),
    ({"pci_bus_id": "42"}, {0: {"pci_bus_id": "43"}, 1: {"pci_bus_id": "42"}},
     {0: 20, 1: 12}),
    ({"uuid": "a", "pci_bus_id": "42"},
     {0: {"uuid": "a", "pci_bus_id": "43"}, 1: {"uuid": "b", "pci_bus_id": "42"}},
     {0: 12, 1: 20}),
    ({"uuid": "a"}, {0: None, 1: {"uuid": "b"}}, {0: 12, 1: 20}),
])
def test_producer_reserve_follows_physical_identity_when_visibility_changes(
        producer, identities, expected):
    from gpuwm.prepared_single_domain_forecast import _devices_stream_budgets

    head = {"decision": {"producer_reserve_bytes": 8, "producer_card": 0,
                         "producer_device_identity": producer}}
    assert _devices_stream_budgets({0: 20, 1: 20}, head, identities=identities) == expected


def test_device_identity_normalizes_uuid_and_pci_without_ordinal(monkeypatch):
    import sys
    from gpuwm.core.device_probe import cuda_device_identity

    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(cuda=SimpleNamespace(
        runtime=SimpleNamespace(getDeviceProperties=lambda _dev: {"uuid": b"\x12" * 16},
                                deviceGetPCIBusId=lambda _dev: b"0000:42:00.0"))))
    # Both identity reads use the complete host-only runtime above. Keep
    # CUDA visibility disabled while testing canonical identity formatting.
    monkeypatch.setattr("gpuwm.local_gpu.no_local_gpu", lambda: False)
    assert cuda_device_identity(7) == {
        "uuid": "12" * 16, "pci_bus_id": "00000000:42:00.0"}


def test_resident_and_tile_plans_keep_the_preparation_reserve():
    from gpuwm.prepared_single_domain_forecast import (
        _stream_reserved_machine, _stream_reserved_options,
    )
    from gpuwm.core.streaming import StreamingOptions
    from tilestream.autoplan import Machine

    head = {"decision": {"producer_reserve_bytes": 8, "producer_card": 0}}
    machine = Machine(vram_bytes=20, host_bytes=100)
    reserved = _stream_reserved_machine(machine, head)
    assert reserved.vram_bytes == 12 and machine.vram_bytes == 20
    assert _stream_reserved_machine(machine, head, device=1).vram_bytes == 12
    capped = _stream_reserved_options(StreamingOptions(mode="auto", vram_budget_bytes=18),
                                       reserved, head)
    assert capped.vram_budget_bytes == 12
    smaller = StreamingOptions(mode="auto", vram_budget_bytes=10)
    assert _stream_reserved_options(smaller, reserved, head).vram_budget_bytes == 10


def test_streamed_rank_price_retains_full_host_boundary_series():
    from dataclasses import replace
    from datetime import datetime
    from gpuwm.config import RunConfig
    from gpuwm.core.devices import DeviceOptions
    from gpuwm.core.devices_memory import estimate_devices
    from gpuwm.experiment import experiment_from_run_config

    run = RunConfig(nx=120, ny=90, nz=12, dx=3000, dy=3000, ztop=12000,
                    dt=15, run_seconds=10800, specified=True)
    exp = replace(experiment_from_run_config(run, datetime(2026, 1, 1)),
                  devices=DeviceOptions(count=2))
    one = estimate_devices(exp, forcing_intervals=1)
    full = estimate_devices(exp, forcing_intervals=3)
    streamed = estimate_devices(exp, forcing_intervals=3, streaming_boundaries=True)
    assert streamed["host_boundary_bytes"] == full["host_boundary_bytes"]
    assert streamed["host_boundary_bytes"] == 3 * one["host_boundary_bytes"]
    assert [card["total_bytes"] for card in streamed["cards"]] == \
        [card["total_bytes"] for card in one["cards"]]


@pytest.mark.parametrize("clock_seconds", [None, 2.0])
def test_ranked_boundary_preflight_uses_domain_clock(clock_seconds):
    asked = []
    run = object.__new__(RankedRun)
    run.boundaries = SimpleNamespace(
        intervals=SimpleNamespace(bounds=((0., 2.), (2., 4.))),
        interval_at=lambda elapsed: asked.append(elapsed))
    run._boundary_clock = (None if clock_seconds is None else
                           SimpleNamespace(elapsed_seconds=clock_seconds))
    run._require_boundary_interval(1.0)
    assert asked == [1.0 if clock_seconds is None else clock_seconds]


@pytest.mark.parametrize("error_type", [boundary_stream.StreamedClockChanged,
                                       boundary_stream.BoundaryProducerFailed])
def test_ranked_boundary_failure_precedes_dispatch(monkeypatch, error_type):
    from gpuwm.core import dycore

    failure = error_type("test boundary failed")
    run = object.__new__(RankedRun)
    run._closed = False
    run.cfg = object()
    run.devices = [0, 1]
    run._clock = {"elapsed_seconds": 2.0}
    run._boundary_clock = None
    run._gather_exposed = lambda: None
    run._order_after_caller = lambda: None

    def fail(_elapsed):
        raise failure

    run.boundaries = SimpleNamespace(
        intervals=SimpleNamespace(bounds=((0., 2.), (2., 4.))), interval_at=fail)
    monkeypatch.setattr(dycore, "begin_wrf_cfl_domain_step", lambda *_a, **_k:
                        pytest.fail("rank work started before its interval was sealed"))
    with pytest.raises(error_type) as caught:
        run.sweep()
    assert caught.value is failure
