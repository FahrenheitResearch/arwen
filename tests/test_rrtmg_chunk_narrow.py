"""#310: the RRTMG batch chunk width is sized to the device, not hardwired.

The legacy-RRTMG batched engines carried fixed column-chunk constants
(SW 2048, LW 4096) sized against a 170 SM part's resident-thread
capacity.  Every card pays that workspace whether or not it can hold the
threads: the chunk transient is mesh-independent (the ledger measured
1,745.6 MiB for the SW spcvmc workspace alone at nz=55) and on smaller
parts most of it buys occupancy the card cannot host.

The remedy is default-on: with no explicit ``column_chunk`` the batched
engines size the chunk to saturate THIS device's resident-thread
capacity, rounded up to a quantum and never below it.  The LW keeps the
old constant as its ceiling.  The SW has no ceiling any more: the 2048
was reasoned against 170 SMs x 1536 threads, and a 170 SM part with
2048 threads per SM saturates at 3328 columns, so the ceiling cost that
part 44 chunks where 27 cover the same columns; the SW width is bounded
by the VRAM free when it is first derived instead
(tests/test_rrtmg_sw_workspace.py covers the bound).  Without a CUDA
device the width is the old constant -- the exact pre-#310 behaviour,
so CPU-only environments and pricing stay deterministic.

Per-column results are bitwise identical at any chunk width (both
translation units' own contract, proved over the full fixture decks by
tests/test_rrtmg_lw_cuda.py and tests/test_rrtmg_sw_cuda.py), so the
sizing moves workspace shape only, never bytes.
"""

import numpy as np
import pytest

from gpuwm.core import rrtmg_lw as _lw
from gpuwm.core import rrtmg_sw as _sw
from gpuwm.core.rrtmg_lw import (BATCH_CHUNK_QUANTUM, batch_column_chunk,
                                 LW_BATCH_COLUMN_CHUNK_CEILING)
from gpuwm.core.rrtmg_sw import (SW_BATCH_COLUMN_CHUNK_NO_DEVICE,
                                 sw_batch_column_chunk)


SEVENTY_SM = 70 * 1536      # RTX 5070 Ti class: 107,520 resident threads
ONE_SEVENTY_SM = 170 * 1536  # 170 SMs at 1536 threads: 261,120
ONE_SEVENTY_SM_2048 = 170 * 2048  # RTX 5090 class: 348,160 resident threads


def test_the_constants_are_the_pre_310_widths():
    assert SW_BATCH_COLUMN_CHUNK_NO_DEVICE == 2048
    assert LW_BATCH_COLUMN_CHUNK_CEILING == 4096
    assert BATCH_CHUNK_QUANTUM == 256


def test_the_width_saturates_and_the_lw_never_exceeds_its_ceiling():
    # 170 SM x 1536 part: SW needs ceil(261120/112)=2332 columns -> 2560,
    # the quantum multiple above (no ceiling binds any more); LW needs
    # ceil(261120/140) = 1866 -> 2048, HALF of the old 4096 at saturated
    # occupancy.
    assert sw_batch_column_chunk(resident_threads=ONE_SEVENTY_SM) == 2560
    assert batch_column_chunk(
        _lw.NGPTLW, LW_BATCH_COLUMN_CHUNK_CEILING,
        resident_threads=ONE_SEVENTY_SM) == 2048
    # 170 SM x 2048 part (the 5090 class): SW ceil(348160/112) = 3109 ->
    # 3328, the width the old 2048 ceiling clamped.
    assert sw_batch_column_chunk(resident_threads=ONE_SEVENTY_SM_2048) == 3328
    # 70 SM part: SW 1024 (ceil(107520/112)=960 -> 1024), LW 768.
    assert sw_batch_column_chunk(resident_threads=SEVENTY_SM) == 1024
    assert batch_column_chunk(
        _lw.NGPTLW, LW_BATCH_COLUMN_CHUNK_CEILING,
        resident_threads=SEVENTY_SM) == 768


def test_saturation_is_actually_reached_or_the_lw_ceiling_binds():
    for cap in (SEVENTY_SM, ONE_SEVENTY_SM, ONE_SEVENTY_SM_2048,
                46 * 1536, 128 * 1536):
        chunk = batch_column_chunk(_lw.NGPTLW, LW_BATCH_COLUMN_CHUNK_CEILING,
                                   resident_threads=cap)
        assert chunk <= LW_BATCH_COLUMN_CHUNK_CEILING
        assert chunk % BATCH_CHUNK_QUANTUM == 0
        if chunk < LW_BATCH_COLUMN_CHUNK_CEILING:
            assert chunk * _lw.NGPTLW >= cap, (
                "a narrowed chunk must still saturate the device")
        sw_chunk = sw_batch_column_chunk(resident_threads=cap)
        assert sw_chunk % BATCH_CHUNK_QUANTUM == 0
        assert sw_chunk * _sw.NGPTSW >= cap, (
            "the SW width must saturate the device: nothing clamps it")
        assert (sw_chunk - BATCH_CHUNK_QUANTUM) * _sw.NGPTSW < cap, (
            "the SW width is the SMALLEST saturating quantum multiple")


def test_no_ceiling_needs_a_device():
    with pytest.raises(ValueError, match="resident_threads"):
        batch_column_chunk(_sw.NGPTSW, None, resident_threads=0)
    assert batch_column_chunk(_sw.NGPTSW, None,
                              resident_threads=ONE_SEVENTY_SM_2048) == 3328


def test_no_device_means_the_old_width(monkeypatch):
    monkeypatch.setattr(_lw, "_device_resident_threads", lambda: None)
    assert sw_batch_column_chunk() == 2048
    assert sw_batch_column_chunk(resident_threads=0) == 2048
    assert batch_column_chunk(
        _lw.NGPTLW, LW_BATCH_COLUMN_CHUNK_CEILING) == 4096


def test_the_floor_holds_for_absurdly_small_capacity():
    assert batch_column_chunk(
        _lw.NGPTLW, LW_BATCH_COLUMN_CHUNK_CEILING,
        resident_threads=1) == BATCH_CHUNK_QUANTUM
    assert sw_batch_column_chunk(resident_threads=1) == BATCH_CHUNK_QUANTUM


def test_the_module_attributes_are_the_device_widths(monkeypatch):
    monkeypatch.setattr(_lw, "_device_resident_threads",
                        lambda: SEVENTY_SM)
    assert _lw.LW_BATCH_COLUMN_CHUNK == 768
    assert _sw.SW_BATCH_COLUMN_CHUNK == 1024
    monkeypatch.setattr(_lw, "_device_resident_threads",
                        lambda: ONE_SEVENTY_SM_2048)
    assert _sw.SW_BATCH_COLUMN_CHUNK == 3328
    monkeypatch.setattr(_lw, "_device_resident_threads", lambda: None)
    assert _lw.LW_BATCH_COLUMN_CHUNK == 4096
    assert _sw.SW_BATCH_COLUMN_CHUNK == 2048


def test_the_legacy_pricing_prices_the_device_width(monkeypatch):
    from gpuwm.core.rrtmg_legacy import legacy_radiation_vram_bytes

    monkeypatch.setattr(_lw, "_device_resident_threads", lambda: None)
    wide = legacy_radiation_vram_bytes(ncol=40962, nz=55, p_top=5000.0)
    monkeypatch.setattr(_lw, "_device_resident_threads",
                        lambda: SEVENTY_SM)
    narrow = legacy_radiation_vram_bytes(ncol=40962, nz=55, p_top=5000.0)
    # The default-on remedy: a bare call on the 70 SM class prices about
    # half the workspace of the hardwired constants.
    assert narrow < wide
    assert narrow <= 0.55 * wide
    # The 5090 class prices the 3328-column SW chunk the device runs,
    # above what the old ceiling priced.
    monkeypatch.setattr(_lw, "_device_resident_threads",
                        lambda: ONE_SEVENTY_SM_2048)
    big = legacy_radiation_vram_bytes(ncol=40962, nz=55, p_top=5000.0)
    assert big > wide
    # An explicit column_chunk is untouched by the sizing.
    pinned = legacy_radiation_vram_bytes(
        ncol=40962, nz=55, p_top=5000.0, column_chunk=2048)
    monkeypatch.setattr(_lw, "_device_resident_threads", lambda: None)
    assert pinned == legacy_radiation_vram_bytes(
        ncol=40962, nz=55, p_top=5000.0, column_chunk=2048)


def test_this_device_width_saturates_this_card():
    cp = pytest.importorskip("cupy")
    try:
        cap = int(cp.cuda.Device().attributes["MultiProcessorCount"]) * \
            int(cp.cuda.Device().attributes["MaxThreadsPerMultiProcessor"])
    except Exception:
        pytest.skip("no usable CUDA device")
    measured = _lw._device_resident_threads()
    assert measured == cap
    lw_chunk = _lw.LW_BATCH_COLUMN_CHUNK
    assert lw_chunk <= LW_BATCH_COLUMN_CHUNK_CEILING
    if lw_chunk < LW_BATCH_COLUMN_CHUNK_CEILING:
        assert lw_chunk * _lw.NGPTLW >= cap
    sw_chunk = _sw.SW_BATCH_COLUMN_CHUNK
    assert sw_chunk * _sw.NGPTSW >= cap
    assert (sw_chunk - BATCH_CHUNK_QUANTUM) * _sw.NGPTSW < cap
