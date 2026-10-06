"""A card reading that decides something does not count this process's garbage as used.

Admission, tile planning and the ensemble pack are all sized from a free
figure read off the card. Arrays reachable only through a reference cycle
(a finished forecast's driver/state attachment is one) stay allocated until
the cyclic collector happens to run, and the default pool keeps the blocks
of earlier work cached; ``cudaMemGetInfo`` counts both as used. The release
gate's one-process GPU shard showed what that costs: after the two
member-sources ensemble tests, the native decline gate read 2.87 GiB free on
a card held to 15 GiB and its first member was refused, and in another run
its four-member native pack was admitted as two packs. Run alone, it passed.

Each test here builds that garbage on purpose, with the collector held off,
and checks every shared reading against a reading taken without it. A
control asserts the garbage really occupies the card first, so a test that
silently stopped making garbage cannot pass.
"""
import gc

import pytest

from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]

_GARBAGE_BYTES = 512 << 20
#: Other tenants of a shared card move its free figure between readings.
_SLACK_BYTES = 96 << 20


def _clean_free():
    import cupy as cp
    gc.collect()
    cp.cuda.Device().synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    return int(cp.cuda.runtime.memGetInfo()[0])


def _make_cycle_garbage():
    """A self-referencing holder of one device array, dropped while the collector is off."""
    import cupy as cp
    holder = {"array": cp.ones(_GARBAGE_BYTES // 4, dtype=cp.float32)}
    holder["self"] = holder
    cp.cuda.Device().synchronize()
    del holder


def _make_cached_blocks():
    import cupy as cp
    array = cp.ones(_GARBAGE_BYTES // 4, dtype=cp.float32)
    cp.cuda.Device().synchronize()
    del array


@pytest.fixture
def collector_off():
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def _readers():
    from gpuwm.core.preflight import device_free_and_total_bytes
    from gpuwm.core.resident_admission import device_free_bytes
    from tilestream.autoplan import Machine
    return {
        "device_free_and_total_bytes": lambda: device_free_and_total_bytes()[0],
        "resident_admission.device_free_bytes": device_free_bytes,
        "Machine.detect": lambda: Machine.detect(host_bytes=1 << 30).vram_bytes,
    }


@pytest.mark.parametrize("reader", sorted(_readers()))
@pytest.mark.parametrize("make", [_make_cycle_garbage, _make_cached_blocks],
                         ids=["cycle-garbage", "cached-blocks"])
def test_a_decision_reading_sees_garbage_as_free(reader, make, collector_off):
    """The reading gains the garbage back, measured against a raw reading just before it.

    Both comparisons are differential, taken milliseconds apart, so another
    tenant of a shared card cannot move them by more than the slack.
    """
    import cupy as cp
    read = _readers()[reader]
    read()                                # first-call module loads happen here, not below
    baseline = _clean_free()
    make()
    raw = int(cp.cuda.runtime.memGetInfo()[0])
    assert raw <= baseline - _GARBAGE_BYTES + _SLACK_BYTES, (
        "control: the garbage this test builds must occupy the card before the reading")
    reading = int(read())
    assert reading >= raw + _GARBAGE_BYTES - _SLACK_BYTES, (
        f"{reader} read {reading / 2**30:.3f} GiB with {_GARBAGE_BYTES >> 20} MiB of this "
        f"process's garbage on the card, where the raw driver figure was {raw / 2**30:.3f} GiB")


def test_the_reclaim_leaves_live_arrays_alone(collector_off):
    import cupy as cp
    from gpuwm.core.preflight import release_unreachable_device_memory
    live = cp.arange(1 << 20, dtype=cp.float32)
    expected = float(live.sum())
    _make_cycle_garbage()
    release_unreachable_device_memory(cp)
    assert float(live.sum()) == expected
    assert cp.get_default_memory_pool().used_bytes() >= live.nbytes
