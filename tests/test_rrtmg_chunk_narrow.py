"""#310: the RRTMG batch chunk width is sized to the device, not hardwired.

The legacy-RRTMG batched engines carried fixed column-chunk constants
(SW 2048, LW 4096) sized against a 170 SM part's resident-thread
capacity.  Every card pays that workspace whether or not it can hold the
threads: the chunk transient is mesh-independent (the ledger measured
1,745.6 MiB for the SW spcvmc workspace alone at nz=55) and on smaller
parts most of it buys occupancy the card cannot host.

The remedy is default-on: with no explicit ``column_chunk`` the batched
engines size the chunk to saturate THIS device's resident-thread
capacity, rounded up to a quantum, never above the old constant (which
becomes the ceiling) and never below the quantum.  Without a CUDA device
the width is the ceiling — the exact pre-#310 behaviour, so CPU-only
environments and pricing stay deterministic and conservative.

Per-column results are bitwise identical at any chunk width (both
translation units' own contract, proved over the full fixture decks by
tests/test_rrtmg_lw_cuda.py and tests/test_rrtmg_sw_cuda.py), so the
narrowing moves workspace shape only, never bytes.
"""

import numpy as np
import pytest

from gpuwm.core import rrtmg_lw as _lw
from gpuwm.core import rrtmg_sw as _sw
from gpuwm.core.rrtmg_lw import (BATCH_CHUNK_QUANTUM, batch_column_chunk,
                                 LW_BATCH_COLUMN_CHUNK_CEILING)
from gpuwm.core.rrtmg_sw import SW_BATCH_COLUMN_CHUNK_CEILING


SEVENTY_SM = 70 * 1536      # RTX 5070 Ti class: 107,520 resident threads
ONE_SEVENTY_SM = 170 * 1536  # RTX 5090 class: 261,120 resident threads


def test_the_ceilings_are_the_pre_310_constants():
    assert SW_BATCH_COLUMN_CHUNK_CEILING == 2048
    assert LW_BATCH_COLUMN_CHUNK_CEILING == 4096
    assert BATCH_CHUNK_QUANTUM == 256


def test_the_width_saturates_and_never_exceeds_the_ceiling():
    # 170 SM part: SW needs ceil(261120/112)=2332 columns -> the 2048
    # ceiling binds (unchanged from today); LW needs ceil(261120/140)
    # = 1866 -> 2048, HALF of the old 4096 at saturated occupancy.
    assert batch_column_chunk(
        _sw.NGPTSW, SW_BATCH_COLUMN_CHUNK_CEILING,
        resident_threads=ONE_SEVENTY_SM) == 2048
    assert batch_column_chunk(
        _lw.NGPTLW, LW_BATCH_COLUMN_CHUNK_CEILING,
        resident_threads=ONE_SEVENTY_SM) == 2048
    # 70 SM part: SW 1024 (ceil(107520/112)=960 -> 1024), LW 768.
    assert batch_column_chunk(
        _sw.NGPTSW, SW_BATCH_COLUMN_CHUNK_CEILING,
        resident_threads=SEVENTY_SM) == 1024
    assert batch_column_chunk(
        _lw.NGPTLW, LW_BATCH_COLUMN_CHUNK_CEILING,
        resident_threads=SEVENTY_SM) == 768


def test_saturation_is_actually_reached_or_the_ceiling_binds():
    for tpc, ceiling in ((_sw.NGPTSW, SW_BATCH_COLUMN_CHUNK_CEILING),
                         (_lw.NGPTLW, LW_BATCH_COLUMN_CHUNK_CEILING)):
        for cap in (SEVENTY_SM, ONE_SEVENTY_SM, 46 * 1536, 128 * 1536):
            chunk = batch_column_chunk(tpc, ceiling, resident_threads=cap)
            assert chunk <= ceiling
            assert chunk % BATCH_CHUNK_QUANTUM == 0
            if chunk < ceiling:
                assert chunk * tpc >= cap, (
                    "a narrowed chunk must still saturate the device")


def test_no_device_means_the_old_width(monkeypatch):
    monkeypatch.setattr(_lw, "_device_resident_threads", lambda: None)
    assert batch_column_chunk(
        _sw.NGPTSW, SW_BATCH_COLUMN_CHUNK_CEILING) == 2048
    assert batch_column_chunk(
        _lw.NGPTLW, LW_BATCH_COLUMN_CHUNK_CEILING) == 4096


def test_the_floor_holds_for_absurdly_small_capacity():
    assert batch_column_chunk(
        _lw.NGPTLW, LW_BATCH_COLUMN_CHUNK_CEILING,
        resident_threads=1) == BATCH_CHUNK_QUANTUM


def test_the_module_attributes_are_the_narrowed_widths(monkeypatch):
    monkeypatch.setattr(_lw, "_device_resident_threads",
                        lambda: SEVENTY_SM)
    assert _lw.LW_BATCH_COLUMN_CHUNK == 768
    assert _sw.SW_BATCH_COLUMN_CHUNK == 1024
    monkeypatch.setattr(_lw, "_device_resident_threads", lambda: None)
    assert _lw.LW_BATCH_COLUMN_CHUNK == 4096
    assert _sw.SW_BATCH_COLUMN_CHUNK == 2048


def test_the_legacy_pricing_prices_the_narrowed_width(monkeypatch):
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
    # An explicit column_chunk is untouched by the narrowing.
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
    for tpc, ceiling, chunk in (
            (_sw.NGPTSW, SW_BATCH_COLUMN_CHUNK_CEILING,
             _sw.SW_BATCH_COLUMN_CHUNK),
            (_lw.NGPTLW, LW_BATCH_COLUMN_CHUNK_CEILING,
             _lw.LW_BATCH_COLUMN_CHUNK)):
        assert chunk <= ceiling
        if chunk < ceiling:
            assert chunk * tpc >= cap
