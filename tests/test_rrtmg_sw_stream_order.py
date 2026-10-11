"""Shared SW tables must be ready before another stream's first kernel."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core import rrtmg_sw as sw


def test_first_kernel_waits_once_per_stream_and_retains_the_stream():
    calls = []

    class Stream:
        def __init__(self, ptr):
            self.ptr = ptr

        def wait_event(self, event):
            calls.append((self.ptr, event))

    current = [Stream(1)]
    ready = object()
    engine = object.__new__(sw.CudaSW)
    engine.cp = SimpleNamespace(cuda=SimpleNamespace(get_current_stream=lambda: current[0]))
    engine.module = SimpleNamespace(get_function=lambda name: name)
    engine._tables_ready = ready
    engine._table_streams = {}
    assert engine._k("rsw_setcoef") == "rsw_setcoef"
    engine._k("rsw_taumol")
    current[0] = Stream(2)
    engine._k("rsw_inatm_layers_b")
    engine._k("rsw_setcoef_b")
    assert calls == [(1, ready), (2, ready)]
    assert engine._table_streams[2] is current[0]


@pytest.mark.gpu
def test_delayed_table_upload_orders_two_member_streams(monkeypatch):
    """Run the real constructor with delayed device copies of its real tables.

    The inputs and zero destinations are ready before the producer delay.
    Only the three constructor uploads are replaced. Each member enters _k
    exactly as the scalar and batched drivers do, then reads every table word.
    Without the constructor event / first-use wait, the reads see zero tables.
    """
    cp = pytest.importorskip("cupy")
    import inspect
    import threading
    import time
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("no CUDA device")
    if not count:
        pytest.skip("no CUDA device")
    fixture = Path(__file__).resolve().parents[1] / "tools" / "rrtmg_wrf461_oracle" / "sw_fixtures" / "sw_tables.npz"
    with np.load(fixture) as data:
        tab = sw.tables_from_dump(dict(data))
    packed, _ = sw._pack_cuda_tables(tab)
    probe = object.__new__(sw.CudaSW)
    probe.tab = tab
    host = [packed, np.asarray(tab.ngb, dtype=np.int32),
            np.asarray(probe._laysolfr_band_spec(), dtype=np.int32)]
    original = cp.asarray
    source = [original(a) for a in host]
    destinations = [cp.zeros_like(a) for a in source]
    outputs = [[cp.empty_like(a) for a in source] for _ in range(2)]
    # Resolve table kernels and both copy dtypes before timing the delay.
    # A cold driver/kernel lookup must not silently consume the race window.
    warm = sw.CudaSW(tab)
    warm._k("rsw_setcoef")
    for destination, initial in zip(destinations, source):
        cp.copyto(destination, initial)
        destination.fill(0)
    for output in outputs:
        for result, initial in zip(output, source):
            cp.copyto(result, initial)
    read = cp.RawKernel(r'''
        extern "C" __global__ void read_bytes(const unsigned char* table,
                                              unsigned char* out,
                                              unsigned long long bytes) {
            unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
            for (; i < bytes; i += gridDim.x * blockDim.x) out[i] = table[i];
        }
    ''', "read_bytes")
    def snapshot(table, result):
        read((128,), (256,), (table, result, np.uint64(table.nbytes)))

    wait = cp.RawKernel(r'''
        extern "C" __global__ void wait_gate(int* flag) {
            while (atomicAdd(flag, 0) == 0) __nanosleep(100);
        }
    ''', "wait_gate")
    release = cp.RawKernel(r'''
        extern "C" __global__ void release_gate(int* flag) { atomicExch(flag, 1); }
    ''', "release_gate")
    gate = cp.ones(1, dtype=cp.int32)
    wait((1,), (1,), (gate,))
    release((1,), (1,), (gate,))
    gate.fill(0)
    for output in outputs:
        for result, table in zip(output, destinations):
            snapshot(table, result)
    cp.cuda.get_current_stream().synchronize()
    producer = cp.cuda.Stream(non_blocking=True)
    members = [cp.cuda.Stream(non_blocking=True) for _ in range(2)]
    releaser = cp.cuda.Stream(non_blocking=True)
    uploads = []
    timed_out = []
    device = int(cp.cuda.Device().id)

    def open_gate():
        with cp.cuda.Device(device), releaser:
            release((1,), (1,), (gate,))

    def watchdog():
        timed_out.append(True)
        open_gate()

    timer = threading.Timer(10, watchdog)

    def delayed_upload(array, *args, **kwargs):
        index = len(uploads)
        assert index < 3, "unexpected constructor upload"
        np.testing.assert_array_equal(array, host[index])
        if index == 0:
            # Resolve this actual module before the gate. Lazy function loading
            # must not synchronize the device and hide a missing event.
            owner = inspect.currentframe().f_back.f_locals['self']
            owner.module.get_function('rsw_setcoef')
            wait((1,), (1,), (gate,))
        # Use a kernel upload as well as a kernel consumer. CUDA memcpy engines
        # can serialize independent queues and make the original race pass.
        snapshot(source[index], destinations[index])
        uploads.append(index)
        return destinations[index]

    try:
        timer.start()
        with monkeypatch.context() as patch:
            patch.setattr(cp, "asarray", delayed_upload)
            with producer:
                engine = sw.CudaSW(tab)
        assert uploads == [0, 1, 2]
        assert not timed_out, "constructor synchronized the delayed producer"
        assert not producer.done, "producer gate released before member dispatch"
        for member, output in zip(members, outputs):
            with member:
                engine._k("rsw_setcoef")
                for result, table in zip(output, destinations):
                    # A CUDA kernel can read while the copy engine waits on
                    # the delayed producer. A D2D memcpy consumer would share
                    # that engine and could hide the missing stream ordering.
                    snapshot(table, result)
        time.sleep(.05)
        assert not timed_out, "member dispatch synchronized the delayed producer"
        assert not producer.done, "producer gate released before member reads"
        open_gate()
        for member in members:
            member.synchronize()
        for output in outputs:
            for result, wanted in zip(output, host):
                np.testing.assert_array_equal(cp.asnumpy(result).view(np.uint8), wanted.view(np.uint8))
    finally:
        timer.cancel()
        open_gate()
        releaser.synchronize()
        producer.synchronize()
        for member in members:
            member.synchronize()


@pytest.mark.gpu
def test_two_member_streams_keep_every_shortwave_output_bitwise(monkeypatch):
    import test_rrtmg_sw_cuda as deck
    from gpuwm.core.deferred_device_checks import deferred

    cp = deck.cp
    engine = sw.CudaSW(deck.tables())
    monkeypatch.setattr(deck, "_cuda", engine)
    groups = deck._deck_groups()
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(2)]
    for cases in groups.values():
        inputs = deck._group_inputs(cases)
        reference = deck._run_batched(cases, inputs, chunk=3)
        # Use the device entry so neither member drains its stream to return
        # host arrays. Launch both before synchronizing either of them.
        results = []
        with deferred():
            for stream in streams:
                with stream:
                    with monkeypatch.context() as patch:
                        patch.setattr(engine, "rrtmg_sw_batched", engine.rrtmg_sw_batched_device)
                        results.append(deck._run_batched(cases, inputs, chunk=3))
            for stream in streams:
                stream.synchronize()
        for member, result in enumerate(results):
            host = sw.sw_batched_to_host(result)
            for key in deck.OUT_KEYS:
                deck.assert_bits(f"member{member}/{key}", host[key], reference[key])
    assert len(engine._table_streams) == 3
    assert engine._scratch_by_stream[int(streams[0].ptr)] is not engine._scratch_by_stream[int(streams[1].ptr)]
