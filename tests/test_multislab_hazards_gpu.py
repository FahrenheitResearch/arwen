"""Same-domain, different-seed step identity across non-blocking streams.

Run directly: python -m tests.test_multislab_hazards_gpu
The full step includes terrain, specified boundaries, radiation, UH and the
microphysics-time reflectivity handoff. No halo exchange is needed because
this detector compares independent slabs against their own serial reference.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def _config(suite, sfclay=91):
    from tilestream.multigpu import forced_config
    base = forced_config(96, 80, 50)
    choices = dict(mp_physics=8, bl_pbl_physics=1,
                   sf_sfclay_physics=sfclay, sf_surface_physics=2,
                   num_soil_layers=4, ra_rrtmg_variant="rte-rrtmgp",
                   cu_physics=0)
    if suite == "city":
        choices.update(mp_physics=28, bl_pbl_physics=5,
                       sf_sfclay_physics=5, sf_surface_physics=3,
                       num_soil_layers=9, aer_init_opt=1, wif_input_opt=1, ra_physics=4,
                       ra_rrtmg_variant="rrtmg_legacy")
    # The suites the split's real-case proof also covers: each
    # changes the scheme whose scratch two slabs could share.
    elif suite == "default":
        choices.update(mp_physics=10, cu_physics=1)
    elif suite == "kf":
        choices.update(cu_physics=1)
    elif suite == "gf":
        choices.update(cu_physics=3)
    elif suite == "noahmp":
        choices.update(sf_surface_physics=4)
    elif suite == "myj":
        choices.update(bl_pbl_physics=2, sf_sfclay_physics=2)
    elif suite == "morrison":
        choices.update(mp_physics=10)
    return replace(base, moist=True, ztop=20000.0, dt=1.0, time_step_sound=4, km_opt=4,
                   diff_6th_opt=2, nwp_diagnostics=1,
                   ra_lw_physics=4, ra_sw_physics=4, radt=0.05,
                   **choices)


def _snapshot(state):
    import cupy as cp
    from tilestream.physics_inventory import carrier_manifest, carrier_scalars
    arrays = dict(carrier_manifest(state))
    arrays.update({"physics/" + k: v for k, v in state.physics.fields.items()
                   if isinstance(v, cp.ndarray)})
    arrays.update({"diagnostic/" + k: v for k, v in state._scratch.items()
                   if k in ("refl_10cm", "up_heli_max", "uh")})
    return ({k: cp.asnumpy(v).view(np.uint8).copy() for k, v in arrays.items()},
            carrier_scalars(state))


def _run(suite, devices, concurrent, sfclay=91):
    import cupy as cp
    from gpuwm.core import dycore
    from gpuwm.core.refl import consume_refl_10cm
    from gpuwm.ingest.lateral_bc import (build_state_lateral_boundaries,
                                        attach_lateral_boundaries)
    from tilestream import harness

    states, configs, streams = [], [], []
    for rank, device in enumerate(devices):
        cfg = _config(suite, sfclay)
        with cp.cuda.Device(device):
            geo = harness.make_geography(cfg, terrain=True, periodic_faces=False)
            state, _ = harness.make_physics_state(cfg, 101 + 101 * rank,
                                                  geography=geo)
            other = harness.make_state(cfg, 102 + 101 * rank, geography=geo)
            boundaries = build_state_lateral_boundaries(
                [state, other], [0.0, 3600.0],
                spec_bdy_width=cfg.spec_bdy_width, spec_zone=cfg.spec_zone,
                relax_zone=cfg.relax_zone)
            attach_lateral_boundaries(state, boundaries)
            cp.cuda.get_current_stream().synchronize()
            states.append(state)
            configs.append(cfg)
            streams.append(cp.cuda.Stream(non_blocking=True))
    barrier = Barrier(2) if concurrent else None

    def work(rank):
        with cp.cuda.Device(devices[rank]), streams[rank]:
            state, cfg = states[rank], configs[rank]
            for _ in range(6):
                if barrier is not None:
                    barrier.wait(timeout=120)
                dycore.step(state, cfg, refl_10cm_due=True)
                consume_refl_10cm(state)
            streams[rank].synchronize()
            assert state.physics.call_counts["radiation"] >= 2
            return _snapshot(state)

    if concurrent:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(work, rank) for rank in range(2)]
            return [f.result() for f in futures]
    return [work(rank) for rank in range(2)]


def _assert_identical(reference, concurrent):
    for rank, ((a, scalars_a), (b, scalars_b)) in enumerate(zip(reference, concurrent)):
        assert a.keys() == b.keys()
        for name in a:
            assert np.array_equal(a[name], b[name]), f"rank {rank}: {name} differs in bits"
        assert scalars_a == scalars_b, f"rank {rank}: carrier clocks differ"


@pytest.mark.parametrize("suite,sfclay", [
    ("lean", 91), ("lean", 1), ("city", 5), ("default", 91), ("kf", 91),
    ("gf", 91), ("noahmp", 91), ("myj", 2), ("morrison", 91)])
@pytest.mark.parametrize("two_devices", [False, True])
def test_multislab_step_identity(suite, sfclay, two_devices):
    import cupy as cp
    if two_devices and cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("two-device identity requires two CUDA devices; this host has one")
    devices = [0, 1] if two_devices else [0, 0]
    reference = _run(suite, devices, False, sfclay)
    concurrent = _run(suite, devices, True, sfclay)
    _assert_identical(reference, concurrent)


def test_shared_validation_words_negative_control():
    """A planted shared word must contaminate the valid slab's readback."""
    import cupy as cp
    from gpuwm.core import mynn_pbl_runtime as runtime
    from gpuwm.core.mynn_pbl_gpu import _flag_mask, _nonfinite

    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(2)]
    shared = cp.zeros(6, cp.int32)
    valid = cp.ones(1024, cp.float32)
    invalid = cp.full(1024, cp.nan, cp.float32)
    cp.cuda.get_current_stream().synchronize()
    # Deterministic witness to the allowed interleaving: slab A's completed
    # scan, slab B's completed scan, then slab A's read of the shared word.
    with streams[0]:
        assert not _flag_mask(_nonfinite(), [valid], shared)[0]
    with streams[1]:
        assert _flag_mask(_nonfinite(), [invalid], shared)[0]
    with streams[0]:
        assert int(shared[0].get()) == 1, "planted race was not detected"
        own_a = runtime._validity_flags()
    with streams[1]:
        own_b = runtime._validity_flags()
    assert own_a.data.ptr != own_b.data.ptr
    with streams[0]:
        assert not _flag_mask(_nonfinite(), [valid], own_a)[0]
    with streams[1]:
        assert _flag_mask(_nonfinite(), [invalid], own_b)[0]
    with streams[0]:
        assert int(own_a[0].get()) == 0


def test_radiation_chunk_scratch_stream_ownership():
    import cupy as cp
    from gpuwm.core.rrtmgp import _chunk_scratch

    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(2)]
    buffers = []
    for rank, stream in enumerate(streams):
        with stream:
            buffer, _ = _chunk_scratch("slab_ownership_probe", (1024,), xp=cp)
            buffer.fill(rank + 1)
            stream.synchronize()
            buffers.append(buffer)
    assert buffers[0].data.ptr != buffers[1].data.ptr
    for rank, stream in enumerate(streams):
        with stream:
            assert np.all(buffers[rank].get() == rank + 1)


@pytest.mark.parametrize("two_devices", [False, True])
def test_uploaded_table_eviction_waits_for_readers(two_devices):
    """Eviction cannot recycle a table still being read on another stream."""
    import cupy as cp
    from gpuwm.core.device_cache import cuda_cache

    if two_devices and cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("cross-card cache eviction requires two CUDA devices")
    evicting_device = 1 if two_devices else 0

    producer = cp.cuda.Stream(non_blocking=True)
    consumer = cp.cuda.Stream(non_blocking=True)
    with cp.cuda.Device(evicting_device):
        evictor = cp.cuda.Stream(non_blocking=True)
    pool = cp.cuda.MemoryPool()

    @cuda_cache(maxsize=1, ready=True)
    def table(value):
        return cp.full(1024, value, cp.int32)

    read_after_delay = cp.RawKernel(r'''
    extern "C" __global__ void delayed_read(const int *input, int *output,
                                            unsigned long long cycles) {
        unsigned long long start = clock64();
        while (clock64() - start < cycles) {}
        output[0] = input[0];
    }
    ''', "delayed_read")
    read_after_delay.compile()
    with cp.cuda.using_allocator(pool.malloc):
        with producer:
            first = table(7)
            pointer = first.data.ptr
            output = cp.empty(1, cp.int32)
        with consumer:
            # A cache hit orders the table's upload before this reader.
            first = table(7)
            read_after_delay((1,), (1,), (first, output, np.uint64(50_000_000)))
        del first
        with cp.cuda.Device(evicting_device), evictor:
            second = table(8)
        with producer:
            recycled = cp.full(1024, 99, cp.int32)
            # Force the memory reuse that exposes a missing reader wait.
            assert recycled.data.ptr == pointer
        consumer.synchronize()
        assert int(output[0].get()) == 7
        producer.synchronize()
        evictor.synchronize()
        table.cache_clear()


if __name__ == "__main__":
    code = pytest.main([__file__, "-q", "-rs"])
    print("MULTISLAB_HAZARDS " + ("PASS" if code == 0 else "FAIL"))
    raise SystemExit(code)
