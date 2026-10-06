"""Small same-card resident rank comparison."""
import pytest
pytestmark = pytest.mark.gpu


def test_same_card_ranked_identity():
    from tilestream.ranks_gate import config, integrate
    integrate(config(nx=96, ny=80, nz=12), mode="threads", nsteps=2)


@pytest.mark.parametrize("km_opt", [2, 4])
@pytest.mark.parametrize("mix_full_fields", [True, False])
def test_coordinate_diffusion_same_card_ranked_identity(km_opt, mix_full_fields):
    from dataclasses import replace
    from tilestream.ranks_gate import config, integrate
    cfg = replace(config(nx=96, ny=80, nz=12), diff_opt=1,
                  km_opt=km_opt, mix_full_fields=mix_full_fields)
    integrate(cfg, mode="threads", nsteps=3)


def test_swint_legacy_radiation_same_card_ranked_identity():
    """swint_opt = 1 with legacy RRTMG: two ranks equal the resident run.

    The swint carrier holds its own latitude/longitude grids; a split must
    gather them per rank (tilestream.driver _SCHEME_GEOGRAPHY) and the
    legacy adapter must interpolate ozone from each rank's live latitude.
    A tree whose split refuses the build, or reads a neutral or stale
    latitude, fails here.
    """
    from dataclasses import replace
    from tilestream.ranks_gate import config, integrate
    cfg = replace(config(nx=96, ny=80, nz=12, rung="mynn"), swint_opt=1)
    result = integrate(cfg, mode="threads", nsteps=8, change_live=False)
    assert result["resident_fires"]["radiation"] >= 2


def test_adaptive_same_card_cfl_and_digest():
    from tilestream.ranks_gate import adaptive_loop
    adaptive_loop(nsteps=4)


def test_lazy_store_roundtrip_and_worker_failure(monkeypatch):
    import cupy as cp
    from gpuwm.core import dycore, streaming
    from gpuwm.core.devices import DeviceOptions
    from tilestream import gather
    from tilestream.ranks import RankedRunError
    from tilestream.ranks_gate import config, fixture, make_ranked
    cfg = config(96, 80, 12)
    state, bundle = fixture(cfg)
    options = DeviceOptions(count=2, ids=(0, 0))
    streamed = make_ranked(bundle, cfg, streaming.ranked_decision(cfg, options),
                            options, None, "threads")
    run = streamed.tiled_run
    counts = []
    original_gather = gather.gather_tile
    def count_gather(*args, **kwargs):
        counts.append(1)
        return original_gather(*args, **kwargs)
    monkeypatch.setattr(gather, "gather_tile", count_gather)
    try:
        progress = []
        for _ in range(2):
            dycore.step(state, cfg)
            run.sweep(1, progress=lambda step, rank, window: progress.append((step, rank, window)))
        assert [(step, rank) for step, rank, _ in progress] == [(0, 0), (0, 1)]*2
        assert all(window is run.specs[rank] for _, rank, window in progress)
        assert counts == [], "unread sweeps must not upload the host store"
        joined = run.store
        joined["state/mup"] += 0.125
        state.mup += cp.float32(0.125)
        dycore.step(state, cfg)
        run.sweep(1)
        assert len(counts) == 2, "each rank must gather an exposed store"
        actual = run.store
        for name, array in streaming.streamed_store_inventory()(state).items():
            assert cp.asnumpy(array).tobytes() == actual[name].tobytes(), name
        original_step = run._step_rank
        def fail_rank(rank, *args):
            if rank == 1:
                raise ValueError("injected worker failure")
            original_step(rank, *args)
        monkeypatch.setattr(run, "_step_rank", fail_rank)
        with pytest.raises(RankedRunError, match="rank 1 card 0 failed: injected worker failure"):
            run.sweep(1)
    finally:
        import weakref
        channels = [weakref.ref(array) for channel in run.channels
                    for array in (channel.send, channel.recv)]
        workers = list(run._workers)
        facts = run.devices_report()
        run.close()
        assert run.devices_report() == facts
        assert all(ref() is None for ref in channels), "closed seam buffers remain owned"
        assert all(not worker.is_alive() for worker in workers)


def test_retired_thread_window_cannot_satisfy_a_new_cfl_step():
    from types import SimpleNamespace
    from gpuwm.core import dycore
    from tilestream import harness
    cfg = harness.make_config(8, 4, 4, grid_id=99)
    window = SimpleNamespace(i0=1, i1=7, ci0=0, j0=1, j1=3, cj0=0)
    dycore.reset_wrf_cfl_recording()
    dycore.enable_wrf_cfl_recording()
    try:
        dycore.begin_wrf_cfl_domain_step(cfg)
        dycore.set_wrf_cfl_tile_window(cfg.grid_id, window)
        dycore.finish_wrf_cfl_domain_step(cfg.grid_id, commit=False)
        dycore.begin_wrf_cfl_domain_step(cfg)
        assert dycore.wrf_cfl_capture_key(cfg.grid_id) == (0, None)
        with pytest.raises(RuntimeError, match="no owned-column window"):
            dycore.record_wrf_vertical_cfl(None, cfg, None)
    finally:
        dycore.finish_wrf_cfl_domain_step(cfg.grid_id, commit=False)
        dycore.reset_wrf_cfl_recording()


def test_build_failure_drains_and_preserves_the_device(monkeypatch):
    from gpuwm.core import streaming
    from gpuwm.core.devices import DeviceOptions
    from tilestream import driver
    from tilestream.ranks import RankedRunError
    from tilestream.ranks_gate import config, fixture, make_ranked
    cfg = config(96, 80, 12)
    _, bundle = fixture(cfg)
    options = DeviceOptions(count=2, ids=(0, 0))
    decision = streaming.ranked_decision(cfg, options)
    original = driver.assert_geography_gathered
    def fail_after_upload(*args, **kwargs):
        raise ValueError("injected geography refusal")
    monkeypatch.setattr(driver, "assert_geography_gathered", fail_after_upload)
    with pytest.raises(RankedRunError, match="rank 0 card 0 build failed: injected geography refusal"):
        make_ranked(bundle, cfg, decision, options, None, "threads")
    monkeypatch.setattr(driver, "assert_geography_gathered", original)
    from tilestream import ranks
    original_thread = ranks.Thread
    threads = []
    class FailedStart(original_thread):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            threads.append(self)
        def start(self):
            if self.name.startswith("rank-1-"):
                raise RuntimeError("injected startup refusal")
            super().start()
    monkeypatch.setattr(ranks, "Thread", FailedStart)
    with pytest.raises(RankedRunError, match="rank 1 card 0 worker startup failed: injected startup refusal"):
        make_ranked(bundle, cfg, decision, options, None, "threads")
    assert all(not worker.is_alive() for worker in threads)
    monkeypatch.setattr(ranks, "Thread", original_thread)
    streamed = make_ranked(bundle, cfg, decision, options, None, "threads")
    try:
        streamed(None, cfg)
        assert streamed.health["nan"] is False
    finally:
        streamed.tiled_run.close()


def test_a_streamed_domain_ends_every_step_on_the_tick_exact_clock(monkeypatch):
    """The store-direct final digest reads the streamed domain's scalars.

    A streamed stepper advances its carrier ``elapsed_seconds`` by dt from
    the REAL kernel-facing image the model loop imposes before the step, so
    unless the loop imposes the integer-tick clock again after the step the
    scalars end each step on a free-running sum.  MEASURED before the fix: a
    3 h HRRR-grid city forecast (fractional adaptive dt) recorded 10800.0
    resident and 10800.0002734375 on every split, so the canonical digests
    differed while every wrfout byte matched.  The double here drifts by a
    millisecond per step, far more than FP32 would, so the check cannot pass
    by rounding luck.  (GPU-marked: execute_experiment's pool trim opens a
    context.)
    """
    import numpy as np
    from gpuwm.core.spawn_runner import SpawnRunner
    from gpuwm.runtime import walk_spawn_legs
    from test_spawn_runner import _CountingCoupler, _CountingPreparer
    from test_spawn_streaming import _FakeStreamed, _case, _model_for

    class Drifting(_FakeStreamed):
        def __call__(self, state, cfg, **kw):
            super().__call__(state, cfg, **kw)
            self.scalars["elapsed_seconds"] += float(cfg.dt) + 1.0e-3

    monkeypatch.setattr("gpuwm.core.dycore.step", lambda *_a, **_k: None)
    dexp = _case()
    runner = SpawnRunner.from_experiment(
        dexp, on_child_built=_CountingPreparer(), array_module=np)
    model = _model_for(dexp, runner)
    parent = Drifting(1)
    born = {}

    def factory(grid_id, node):
        born[int(grid_id)] = Drifting(grid_id)
        return born[int(grid_id)]

    walk_spawn_legs(model, dexp, None, spawn_runner=runner,
                    writers=None, lbc_interval_s=None,
                    coupler_factory=_CountingCoupler,
                    spawned_stepper_factory=factory,
                    validate_state=False, steppers={1: parent})
    assert parent.calls == 4 and born[2].calls == 6
    for grid_id, domain in ((1, parent), (2, born[2])):
        assert domain.scalars["elapsed_seconds"] == 240.0, (
            f"d{grid_id:02d} ended on {domain.scalars['elapsed_seconds']!r}, "
            "not the tick-exact 240.0 its state holds")
        assert model.node(grid_id).state.elapsed_seconds == 240.0


@pytest.mark.parametrize("grid", [(1, 2), (2, 2)])
@pytest.mark.parametrize("buffered", [True, False])
def test_output_road_frames_without_a_drain(grid, buffered):
    """A frame's members reach the host without draining the store.

    The ranked output road (``RankedRun.download``): the named members of
    every slab land in the host store on a stream of their own, equal to the
    resident state at that step, with no whole-store drain and no whole-store
    copy back; a reset on the slabs (``zero_scratch``) equals the same reset
    resident.  MEASURED before it, on two RTX PRO 6000s at 3 km CONUS: each
    frame drained the store twice and copied it back twice, 2.66 s of
    stepping lost per frame against 0.25 s on one card.
    """
    import cupy as cp
    from gpuwm.core import dycore, streaming
    from gpuwm.core.devices import DeviceOptions
    from tilestream.ranks_gate import config, fixture
    cfg = config(96, 80, 12)
    state, bundle = fixture(cfg)
    count = grid[0] * grid[1]
    options = DeviceOptions(count=count, grid=grid, ids=(0,) * count)
    streamed = streaming.ranked_domain_builder(
        bundle, clock=None, options=options,
        snapshot_limits=None if buffered else (1,) * count)(
            None, cfg, streaming.ranked_decision(cfg, options))
    run = streamed.tiled_run
    take = streaming.streamed_store_inventory()
    keys = ["state/thp", "state/u", "state/p"]
    reset = "state/qv"
    try:
        for step in range(4):
            dycore.step(state, cfg)
            run.sweep(1)
            if step % 2 == 1:
                assert run.download(keys) == keys
                assert run.download(keys) == [], "a fresh member is not copied twice"
                run.wait_downloads(run.pending_downloads())
                live = take(state)
                for key in keys:
                    assert cp.asnumpy(live[key]).tobytes() == run.raw_store[key].tobytes(), key
                run.zero_scratch(reset)
                take(state)[reset].fill(0)
        report = run.output_report
        assert report["full_drains"] == 0 and report["full_gathers"] == 0
        assert report["frame_downloads"] == 2 and report["scratch_zeroes"] == 2
        if buffered:
            assert report["fallback_downloads"] == 0
            assert 0 < report["frame_snapshot_bytes"] <= count * run._snapshot_limit
        else:
            assert report["fallback_downloads"] == count * 2
            assert report["frame_snapshot_bytes"] == 0
        joined = run.store
        for name, array in take(state).items():
            assert cp.asnumpy(array).tobytes() == joined[name].tobytes(), name
    finally:
        run.close()


@pytest.mark.parametrize("grid", [(1, 2), (2, 2)])
def test_zero_snapshot_policy_fences_even_a_small_requested_subset(grid, monkeypatch):
    """A rank priced without snapshots cannot allocate one for a small frame.

    Force the shared policy's oversized-inventory decision on a small real
    domain. The requested subset fits the unchanged performance cap, so the
    old request-only decision would allocate storage and fail this test.
    Stepping immediately after the download exercises its producer fence.
    """
    import cupy as cp
    from gpuwm.core import devices_memory, dycore, streaming
    from gpuwm.core.devices import DeviceOptions
    from tilestream.ranks_gate import config, fixture, make_ranked

    cfg = config(96, 80, 12)
    state, bundle = fixture(cfg)
    count = grid[0] * grid[1]
    options = DeviceOptions(count=count, grid=grid, ids=(0,) * count)
    monkeypatch.setattr(devices_memory, "frame_snapshot_budget", lambda _cfg: 0)
    streamed = make_ranked(bundle, cfg, streaming.ranked_decision(cfg, options),
                           options, None, "threads")
    run = streamed.tiled_run
    take = streaming.streamed_store_inventory()
    keys = ["state/thp"]
    try:
        assert run._snapshot_budgets == [0] * count
        assert run._snapshot_limit == devices_memory.FRAME_SNAPSHOT_LIMIT_BYTES
        assert all(0 < take(tile, keys)[keys[0]].nbytes < run._snapshot_limit
                   for tile in run.tiles)
        for _ in range(2):
            dycore.step(state, cfg)
            run.sweep(1)
            expected = cp.asnumpy(take(state)[keys[0]]).tobytes()
            assert run.download(keys) == keys
            pending = run.pending_downloads()
            # No host wait before issuing the next numerical step.
            run.sweep(1)
            dycore.step(state, cfg)
            run.wait_downloads(pending)
            assert run.raw_store[keys[0]].tobytes() == expected
        report = run.output_report
        assert report["frame_snapshot_bytes"] == 0
        assert all(not snapshots for snapshots in run._frame_snapshots)
        assert report["fallback_downloads"] == 2 * count
        assert report["full_drains"] == 0 and report["full_gathers"] == 0
        joined = run.store
        for name, array in take(state).items():
            assert cp.asnumpy(array).tobytes() == joined[name].tobytes(), name
    finally:
        run.close()


@pytest.mark.parametrize("grid", [(1, 2), (2, 2)])
def test_output_snapshot_survives_steps_during_a_delayed_download(grid, monkeypatch):
    """Delayed host DMA must neither freeze stepping nor read a later state."""
    import cupy as cp
    from threading import Event, Timer
    from gpuwm.core import dycore, streaming
    from gpuwm.core.devices import DeviceOptions
    from tilestream import gather
    from tilestream.ranks_gate import config, fixture, make_ranked

    cfg = config(96, 80, 12)
    state, bundle = fixture(cfg)
    count = grid[0] * grid[1]
    options = DeviceOptions(count=count, grid=grid, ids=(0,) * count)
    streamed = make_ranked(bundle, cfg, streaming.ranked_decision(cfg, options),
                           options, None, "threads")
    run = streamed.tiled_run
    take = streaming.streamed_store_inventory()
    keys = ["state/thp", "state/u", "state/p"]
    release = Event()
    timer = Timer(2., release.set)
    original = gather.TilePlan.execute
    delayed = False

    def execute(plan, src, dst, stream=None):
        if delayed and plan.direction == "scatter":
            # Delay only this stream's host transfer. A long GPU kernel
            # also blocks unrelated numerical kernels on some hardware.
            stream.launch_host_func(lambda signal: signal.wait(), release)
        return original(plan, src, dst, stream)

    monkeypatch.setattr(gather.TilePlan, "execute", execute)
    try:
        # Warm scratch and snapshot allocations before delaying DMA.
        # cudaMalloc may synchronize the card on a first allocation.
        for _ in range(4):
            dycore.step(state, cfg)
            run.sweep(1)
        run.download(keys)
        run.wait_downloads(run.pending_downloads())
        dycore.step(state, cfg)
        expected = {key: cp.asnumpy(take(state)[key]).tobytes() for key in keys}
        run.sweep(1)
        delayed = True
        timer.start()
        run.download(keys)
        pending = run.pending_downloads()
        delayed = False
        overlaps = []
        exchange = run.exchange_events

        def observe_numerics():
            # Check completed numerical work before its halo transfers.
            # On a shared card those transfers can share the DMA engine
            # with output, even when numerical kernels overlap the copy.
            for dev, stream in zip(run.devices, run.compute_streams):
                with cp.cuda.Device(dev):
                    stepped = cp.cuda.Event(disable_timing=True)
                    stepped.record(stream)
                    stepped.synchronize()
                    overlaps.append(any(not event.done for entry in pending
                                        for _, event in entry["events"]))
            return exchange()

        monkeypatch.setattr(run, "exchange_events", observe_numerics)
        # No reference step or device-wide synchronization here: either
        # would hide a compute stream waiting for the host transfer.
        run.sweep(1)
        if count == 2:
            assert any(overlaps), "numerical stepping waited for the delayed host copy"
        # Four slabs on one card share the DMA engine with their two-round
        # halo exchange and can finish later. The delayed frame must still
        # contain the earlier state, regardless of that scheduling.
        monkeypatch.setattr(run, "exchange_events", exchange)
        run.wait_downloads(pending)
        for key in keys:
            assert run.raw_store[key].tobytes() == expected[key], key
        dycore.step(state, cfg)
        # Reusing the snapshot at the next output must carry the new time.
        run.download(keys)
        run.wait_downloads(run.pending_downloads())
        for key in keys:
            assert run.raw_store[key].tobytes() == cp.asnumpy(take(state)[key]).tobytes(), key
        assert run.output_report["full_drains"] == 0
    finally:
        release.set()
        timer.cancel()
        if timer.ident is not None:
            timer.join()
        run.close()


def test_output_snapshot_rejects_a_strided_source(monkeypatch):
    from gpuwm.core import streaming
    from gpuwm.core.devices import DeviceOptions
    from tilestream.ranks import RankedRunError
    from tilestream.ranks_gate import config, fixture, make_ranked
    cfg = config(96, 80, 12)
    _, bundle = fixture(cfg)
    options = DeviceOptions(count=2, ids=(0, 0))
    owner = make_ranked(bundle, cfg, streaming.ranked_decision(cfg, options),
                        options, None, "threads")
    run = owner.tiled_run
    original = run.inventory_fn

    def strided(state, names=None):
        arrays = original(state, names)
        if state is run.tiles[0] and "state/thp" in arrays:
            arrays["state/thp"] = arrays["state/thp"][:, ::-1, :]
        return arrays

    try:
        run.sweep(1)
        with monkeypatch.context() as patch:
            patch.setattr(run, "inventory_fn", strided)
            with pytest.raises(RankedRunError, match="not C-contiguous"):
                run.download(["state/thp"])
        assert run.output_report["frame_downloads"] == 0
    finally:
        run.close()


@pytest.mark.parametrize("grid", [(1, 2), (2, 2)])
def test_a_slow_outbound_pack_cannot_see_the_next_step(grid, monkeypatch):
    """A slab's next step waits for its OWN outbound packs.

    The seam exchange made a slab's next step wait for the unpacks INTO it
    and not for the packs FROM it, and a pack reads the interior band the
    next step rewrites in place.  MEASURED on the merge-base engine: one
    card split 2x2, HRRR West Texas lean suite, 3 h, 3 of 6 runs on an RTX
    5090 wrote frames that differ from the unsplit run (first differing step
    19 to 41).  Here every pack is queued behind about 50 ms of other work on
    its copy stream, so a next step that does not wait for it rewrites the
    band before the pack reads it, every time; the slabs must still equal
    the unsplit domain bit for bit.
    """
    import cupy as cp
    from gpuwm.core import dycore, streaming
    from gpuwm.core.devices import DeviceOptions
    from tilestream import multigpu
    from tilestream.ranks_gate import config, fixture, make_ranked
    cfg = config(96, 80, 12)
    state, bundle = fixture(cfg)
    count = grid[0] * grid[1]
    options = DeviceOptions(count=count, grid=grid, ids=(0,) * count)
    streamed = make_ranked(bundle, cfg, streaming.ranked_decision(cfg, options),
                           options, None, "threads")
    run = streamed.tiled_run
    ballast = cp.ones((4096, 4096), dtype=cp.float32)
    original = multigpu._SeamChannel.pack

    def slow_pack(self, src_arrays, stream_ptr):
        with cp.cuda.ExternalStream(stream_ptr, cp.cuda.Device().id):
            for _ in range(40):
                ballast @ ballast
        original(self, src_arrays, stream_ptr)

    monkeypatch.setattr(multigpu._SeamChannel, "pack", slow_pack)
    take = streaming.streamed_store_inventory()
    try:
        # The slabs step back to back, with no resident step between their
        # sweeps: a resident step synchronizes the device, which would let
        # every delayed pack finish before the next sweep and hide the race.
        for _ in range(4):
            run.sweep(1)
        for _ in range(4):
            dycore.step(state, cfg)
        joined = run.store
        for name, array in take(state).items():
            assert cp.asnumpy(array).tobytes() == joined[name].tobytes(), name
    finally:
        run.close()
