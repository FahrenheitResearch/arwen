"""Delayed forcing intervals preserve every ranked carrier at each step."""
import pytest

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("ranks", [1, 2, 4])
def test_posted_boundary_rank_histories_are_byte_identical(ranks):
    import os
    import threading
    from dataclasses import replace
    from types import SimpleNamespace

    import cupy as cp
    from gpuwm.core import dycore, streaming
    from gpuwm.core.clock import DomainClock
    from gpuwm.core.devices import DeviceOptions
    from gpuwm.ingest.lateral_bc import (
        attach_lateral_boundaries, bind_lateral_boundary_clock,
    )
    from tilestream.ranks_gate import (
        _SingleCardGateOptions, _clock, config, fixture, make_ranked,
    )
    from tilestream import gather

    available = cp.cuda.runtime.getDeviceCount()
    if os.environ.get("GPUWM_STREAM_GATE_REQUIRE_CARDS") == "1" and available < ranks:
        pytest.skip(f"requires {ranks} physical cards, found {available}")
    ids = tuple(index % available for index in range(ranks))
    cp.cuda.Device(ids[0]).use()
    cfg = replace(config(128, 104, 12), use_adaptive_time_step=False)
    state, bundle = fixture(cfg)
    first = replace(bundle.boundaries.intervals[0], end_seconds=2.0)
    second = replace(first, start_seconds=2.0, end_seconds=4.0, fields={
        name: replace(field, **{
            side_name: replace(side, value=side.value + 2.0 * side.tendency,
                               tendency=side.tendency * 1.25)
            for side_name in ("west", "east", "south", "north")
            for side in (getattr(field, side_name),)})
        for name, field in first.fields.items()})
    eager = replace(bundle.boundaries, intervals=(first, second))
    attach_lateral_boundaries(state, eager)

    class PostedIntervals:
        bounds = ((0.0, 2.0), (2.0, 4.0))
        ready = 1
        reads = []

        def __len__(self):
            return 2

        def __getitem__(self, index):
            if isinstance(index, slice):
                return tuple(self[i] for i in range(*index.indices(2)))
            if index < 0:
                index += 2
            if not 0 <= index < 2:
                raise IndexError(index)
            assert index < self.ready, "read an interval before its marker was posted"
            self.reads.append((index, threading.current_thread().name))
            return (first, second)[index]

    posted = PostedIntervals()
    lazy = replace(eager, intervals=posted)
    options = (_SingleCardGateOptions(ids[0]) if ranks == 1 else
               DeviceOptions(count=ranks, ids=ids, grid=(1, 2) if ranks == 2 else (2, 2)))
    clocks = [DomainClock(replace(_clock(cfg).spec, lbc_interval_ticks=200),
                          100, 400) for _ in range(3)]
    bind_lateral_boundary_clock(state, clocks[0])
    streams = []
    try:
        for boundaries, clock in zip((eager, lazy), clocks[1:]):
            candidate = SimpleNamespace(**vars(bundle))
            candidate.boundaries = boundaries
            candidate.scalars = dict(bundle.scalars)
            candidate.store = {key: gather.pinned_empty(array.shape, array.dtype)
                               for key, array in bundle.store.items()}
            for key, array in bundle.store.items():
                candidate.store[key][...] = array
            decision = streaming.ranked_decision(cfg, options)
            streams.append(make_ranked(candidate, cfg, decision, options, clock, "threads"))
        assert {index for index, _ in posted.reads} == {0}
        for step in range(4):
            if step == 2:
                assert all(index == 0 for index, _ in posted.reads)
                posted.ready = 2
            for clock in clocks:
                if clock.lbc_reset_due():
                    clock.mark_force()
                clock.prepare_step()
            dycore.step(state, cfg)
            for stream in streams:
                stream(None, cfg)
            for clock in clocks:
                clock.advance()
            expected = {key: cp.asnumpy(array)
                        for key, array in streaming.streamed_store_inventory()(state).items()}
            for stream in streams:
                actual = stream.tiled_run.store
                assert actual.keys() == expected.keys()
                assert all(actual[key].tobytes() == expected[key].tobytes()
                           for key in expected), f"rank count {ranks}, step {step + 1}"
                stream.tiled_run._exposed = False
        assert next(thread for index, thread in posted.reads if index == 1) == \
            threading.current_thread().name
        print({"ranks": ranks, "physical_cards": sorted(set(ids)),
               "history_frames": 4, "carriers": len(expected),
               "delayed_interval": 1, "byte_identical": True})
    finally:
        for stream in streams:
            stream.tiled_run.close()
        dycore.reset_wrf_cfl_recording()
