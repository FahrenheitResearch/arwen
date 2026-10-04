"""History staging must own its CUDA card on the caller and writer threads.

The first frame is submitted with its own card current, then consumed by a
new worker thread whose initial card is 0. Before the fix a card-1 writer
failed there, and its store guard reported the failure on the next download.
The second frame starts with caller card 0 to cover the staging side too.
"""

from dataclasses import replace
from datetime import datetime, timedelta
import json

import numpy as np
import pytest


pytestmark = pytest.mark.gpu

_CARDS = [
    pytest.param((1,), id="card1"),
    # Two logical ranks on card 1 keep the nonzero-card guard test runnable
    # on a two-card host. The next two cases use distinct physical cards.
    pytest.param((1, 1), id="ranks1-1"),
    pytest.param((1, 2), id="cards1-2"),
    pytest.param((2, 3), id="cards2-3"),
    pytest.param((3,), id="card3"),
    pytest.param((0, 2), id="cards0-2"),
]


def _reference_frame(state):
    import cupy as cp
    from gpuwm.io.wrfout import _device_state_frame

    with cp.cuda.Device(state.mup.device.id):
        return {name: cp.asnumpy(value) for name, value in
                _device_state_frame(state).items()}


def _check_file(path, expected):
    import netCDF4

    with netCDF4.Dataset(path) as dataset:
        assert dataset.data_model == "NETCDF3_64BIT_OFFSET"
        dataset.set_auto_mask(False)
        for name, value in expected.items():
            actual = np.asarray(dataset.variables[name][0])
            assert actual.shape == value.shape, name
            assert actual.dtype == value.dtype, name
            assert actual.tobytes() == value.tobytes(), name


def test_resident_frame_plan_uses_state_card():
    """Prevent derived history buffers from being allocated on caller card 0."""
    import cupy as cp
    from gpuwm.io.wrfout import _device_state_frame
    from tilestream import harness, output

    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("the state-card test needs two visible CUDA cards")
    caller = cp.cuda.Device().id
    try:
        with cp.cuda.Device(1):
            state = harness.make_state(harness.make_config(12, 10, 4))
            expected = _reference_frame(state)
            expected_plan = output.frame_plan(state)
            theta = cp.asnumpy(state.thp)
        cp.cuda.Device(0).use()
        fields = _device_state_frame(state)
        assert cp.cuda.Device().id == 0
        assert {value.device.id for value in fields.values()
                if isinstance(value, cp.ndarray)} == {1}
        assert output.frame_plan(state) == expected_plan
        assert cp.cuda.Device().id == 0
        assert output._host(state.thp).tobytes() == theta.tobytes()
        assert cp.cuda.Device().id == 0
        for name, value in fields.items():
            assert output._host(value).tobytes() == expected[name].tobytes(), name
    finally:
        cp.cuda.Device(caller).use()


def test_writer_uses_the_nondefault_producer_stream_on_its_card(tmp_path):
    """Prevent staging before a card-1 producer queued on a side stream."""
    import cupy as cp
    from threading import Event, Timer
    from gpuwm.config import soil_layer_count
    from gpuwm.io.wrfout import AsyncDomainWrfoutWriter
    from tilestream import harness

    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("the producer-stream test needs two visible CUDA cards")
    caller = cp.cuda.Device().id
    writer = prior_stream = None
    release = Event()
    timer = Timer(0.25, release.set)
    try:
        with cp.cuda.Device(1):
            cfg = harness.make_config(12, 10, 4)
            state = harness.make_state(cfg)
            expected = _reference_frame(state)
            expected["MU"].fill(83)
            prior_stream = cp.cuda.get_current_stream()
            producer = cp.cuda.Stream(non_blocking=True)
            producer.use()

        cp.cuda.Device(0).use()
        writer = AsyncDomainWrfoutWriter(
            nx=cfg.nx, ny=cfg.ny, nz=cfg.nz, dx=cfg.dx, dy=cfg.dy,
            title="producer stream regression", global_attrs={},
            soil_layers=soil_layer_count(cfg), device=1)
        assert cp.cuda.Device().id == 0
        assert writer.stream.device_id == 1

        with cp.cuda.Device(1):
            assert cp.cuda.get_current_stream().ptr == producer.ptr
            producer.launch_host_func(lambda signal: signal.wait(), release)
            state.mup.fill(83)
        timer.start()
        path = tmp_path / "wrfout_d01_producer_stream"
        writer.submit(path, datetime(2026, 1, 1), state)
        assert cp.cuda.Device().id == 0
        with cp.cuda.Device(1):
            assert cp.cuda.get_current_stream().ptr == producer.ptr
            # The producer waits for the snapshot before its next mutation.
            state.mup.fill(99)
        writer.drain()
        assert cp.cuda.Device().id == 0
        _check_file(path, expected)
        with cp.cuda.Device(1):
            assert np.all(cp.asnumpy(state.mup) == 99)
    finally:
        release.set()
        timer.cancel()
        if timer.ident is not None:
            timer.join()
        try:
            if writer is not None:
                writer.close()
        finally:
            if prior_stream is not None:
                with cp.cuda.Device(1):
                    prior_stream.use()
            cp.cuda.Device(caller).use()


def test_streamed_store_copies_use_the_array_card():
    """Prevent host refreshes and window copies from staging on card 0."""
    import cupy as cp
    from types import SimpleNamespace
    from gpuwm.core import streaming
    from tilestream import harness

    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("the store-copy test needs two visible CUDA cards")
    caller = cp.cuda.Device().id
    try:
        with cp.cuda.Device(1):
            state = harness.make_state(harness.make_config(12, 10, 4))
            state.mup.fill(0)
        host = np.arange(120, dtype=np.float32).reshape(10, 12)
        store = {"state/mup": host.copy()}
        streamed = streaming.StreamedDomain(
            SimpleNamespace(store=store), None, state=state, host_store=True)
        streaming.publish_store(state, streamed)
        cp.cuda.Device(0).use()

        for window in (None, (2, 7, 3, 9)):
            slices = streaming.window_slices(host.shape, window)
            expected = host[slices]
            nbytes = expected.nbytes
            assert streaming.refresh_from_store(
                state, ("mup",), window=window) == nbytes
            assert cp.cuda.Device().id == 0
            with cp.cuda.Device(1):
                assert cp.asnumpy(state.mup[slices]).tobytes() == expected.tobytes()
                state.mup[slices] += cp.float32(1)
            assert streaming.commit_to_store(
                state, ("mup",), window=window) == nbytes
            assert cp.cuda.Device().id == 0
            assert store["state/mup"][slices].tobytes() == (expected + 1).tobytes()

            store["state/mup"][slices] = expected + 2
            assert streamed.sync_to_state(("state/mup",), window=window) == 1
            assert cp.cuda.Device().id == 0
            with cp.cuda.Device(1):
                assert cp.asnumpy(state.mup[slices]).tobytes() == (expected + 2).tobytes()
                state.mup[slices] += cp.float32(1)
            assert streamed.sync_from_state(("state/mup",), window=window) == 1
            assert cp.cuda.Device().id == 0
            assert store["state/mup"][slices].tobytes() == (expected + 3).tobytes()
            store["state/mup"][...] = host

        staged = streaming._to(state.mup, host)
        assert staged.device.id == 1
        assert cp.cuda.Device().id == 0
        assert streaming._to(host, staged).tobytes() == host.tobytes()
        assert cp.cuda.Device().id == 0
        assert streamed.publish(("state/mup",)) == ("state/mup",)
        assert cp.cuda.Device().id == 0
        with cp.cuda.Device(1):
            state.mup += cp.float32(1)
        assert streamed.adopt(("state/mup",)) == ("state/mup",)
        assert cp.cuda.Device().id == 0
        assert store["state/mup"].tobytes() == (host + 1).tobytes()
    finally:
        cp.cuda.Device(caller).use()


def test_tiled_sweep_and_drains_use_the_construction_card():
    """Prevent a card-1 tiled run from entering its streams on card 0."""
    import cupy as cp
    from gpuwm.core import dycore
    from tilestream import driver, gather, harness

    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("the tiled ownership test needs two visible CUDA cards")
    caller = cp.cuda.Device().id
    run = None
    try:
        with cp.cuda.Device(1):
            cfg = harness.make_config(64, 48, 8, dt=1.0, time_step_sound=4)
            state = harness.make_state(cfg)
            store = {name: gather.pinned_copy(cp.asnumpy(value))
                     for name, value in harness.state_arrays(state).items()}
            run = driver.TiledRun(
                store, cfg, 32, 24, harness.halo_radius(cfg), 2,
                periodic=True, write_mode="ring")
            dycore.step(state, cfg)
            expected = {name: cp.asnumpy(value)
                        for name, value in harness.state_arrays(state).items()}
        cp.cuda.Device(0).use()
        run.sweep(1)
        assert cp.cuda.Device().id == 0
        assert run._pending, "the test must exercise deferred stream waits"
        run.sync_compute()
        assert cp.cuda.Device().id == 0
        run.drain()
        assert cp.cuda.Device().id == 0
        for name, value in expected.items():
            assert run.store[name].tobytes() == value.tobytes(), name
        run.close()
        assert cp.cuda.Device().id == 0
    finally:
        try:
            if run is not None:
                run.close()
        finally:
            cp.cuda.Device(caller).use()


@pytest.mark.parametrize("cards", _CARDS)
@pytest.mark.parametrize("route", [
    "resident", "store-direct", "store-direct-unbuffered",
])
def test_history_writer_and_store_guard_own_their_cards(tmp_path, cards, route):
    """Prevent first-history failure when the run does not begin on card 0.

    A real resident reference and a real RankedRun each take two timesteps.
    Native files must retain every reference field bit for bit. Store-direct
    frames are assembled by the worker after real per-rank downloads, then
    the next download reaches the same staging guard as reflectivity output.
    """
    import cupy as cp
    from gpuwm.config import soil_layer_count
    from gpuwm.core import dycore, streaming
    from gpuwm.core.devices import DeviceOptions
    from gpuwm.io import nc_writer_bridge
    from gpuwm.io.wrfout import AsyncDomainWrfoutWriter
    from tilestream.multigpu import forced_config
    from tilestream.ranks_gate import (
        _SingleCardGateOptions, fixture, make_ranked,
    )

    visible = cp.cuda.runtime.getDeviceCount()
    if max(cards) >= visible:
        pytest.skip(f"cards {cards} need {max(cards) + 1} visible CUDA cards")
    reason = nc_writer_bridge.unavailable_reason()
    assert reason is None, f"native history writer unavailable: {reason}"
    caller = cp.cuda.Device().id
    run = writer = None
    try:
        with cp.cuda.Device(cards[0]):
            cfg = replace(forced_config(96, 80, 8), dt=1.0,
                          time_step_sound=4)
            state, bundle = fixture(cfg)
            options = (_SingleCardGateOptions(cards[0]) if len(cards) == 1
                       else DeviceOptions(count=len(cards), ids=cards))
            streamed = make_ranked(
                bundle, cfg, streaming.ranked_decision(cfg, options),
                options, None, "threads")
            run = streamed.tiled_run
            if route == "store-direct-unbuffered":
                run._snapshot_limit = 1
            writer = AsyncDomainWrfoutWriter(
                nx=cfg.nx, ny=cfg.ny, nz=cfg.nz, dx=cfg.dx, dy=cfg.dy,
                title="history device regression", global_attrs={},
                soil_layers=soil_layer_count(cfg))

        expected = []
        paths = []
        start = datetime(2026, 1, 1)
        cp.cuda.Device(0).use()
        for index in range(2):
            with cp.cuda.Device(cards[0]):
                dycore.step(state, cfg)
            run.sweep(1)
            assert cp.cuda.Device().id == 0
            expected.append(_reference_frame(state))

            if index:
                # The prior frame borrows the pinned store. This invokes
                # the real writer guard before any new bytes overwrite it.
                assert run.download(("state/thp",)) == ["state/thp"]
                assert cp.cuda.Device().id == 0

            path = tmp_path / f"wrfout_d01_frame{index}"
            paths.append(path)
            valid = start + timedelta(seconds=index + 1)

            def submit():
                if route == "resident":
                    writer.submit(path, valid, state)
                else:
                    frame = streamed.history_fields()
                    assert frame.deferred
                    writer.submit(path, valid, None, frame=frame)

            if index == 0:
                # Keep the negative control focused on the worker: it has
                # never selected this card even though the producer has.
                with cp.cuda.Device(cards[0]):
                    submit()
            else:
                submit()
            assert cp.cuda.Device().id == 0
            streamed.add_store_guard(writer.drain_staging)

        writer.drain()
        assert cp.cuda.Device().id == 0
        assert writer.paths == paths
        assert len(writer.completed_records) == 2
        for path, frame in zip(paths, expected):
            _check_file(path, frame)

        # Per-domain writers also receive this host-only facade. The real
        # ranked owner, not a NumPy field or caller card, selects its stream.
        from gpuwm.core.streamed_state import CanonicalStoreState
        from gpuwm.io.wrfout import _domain_output_device
        from tilestream import driver
        with cp.cuda.Device(cards[0]):
            shell = CanonicalStoreState(
                bundle.template, cfg, store=run.raw_store,
                geography=bundle.geography, scalars=bundle.scalars,
                inventory=streaming.streamed_store_inventory()(bundle.template),
                geography_inventory=driver.geography_inventory(bundle.template))
        shell._streamed_domain = streamed
        assert shell.mup is run.raw_store["state/mup"]
        assert isinstance(shell.mup, np.ndarray)
        assert _domain_output_device(shell) == cards[0]
        assert cp.cuda.Device().id == 0

        if route != "resident":
            assert run.output_report["full_drains"] == 0
            assert run.output_report["frame_downloads"] >= 2
            if route == "store-direct-unbuffered":
                assert run.output_report["fallback_downloads"] >= len(cards) * 2
        print("HISTORY_DEVICE " + json.dumps({
            "cards": cards, "route": route, "frames": 2,
            "fields": len(expected[0]), "caller_card": cp.cuda.Device().id,
            "worker_stream_card": writer.stream.device_id,
            "files": [str(path) for path in paths],
        }), flush=True)
    finally:
        try:
            if writer is not None:
                writer.close()
        finally:
            try:
                if run is not None:
                    run.close()
            finally:
                cp.cuda.Device(caller).use()
