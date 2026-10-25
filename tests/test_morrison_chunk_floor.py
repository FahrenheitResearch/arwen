"""Morrison's forecast calls: one launch for a small domain, chunks for a big one.

THE BREAKAGE THIS PREVENTS (WOOF 1.1 final canary, the u2 user shape): a
156 x 136 x 50 domain ran its Morrison microphysics as 21 calls of ~1,010
columns (ceil(ncol / NWORK)), three kernels each, 12.7 ms a step on the card
against 0.37 ms for 2.8.7's single launch, and the 48 h forecast took 1.24x
its 2.8.7 twin. The chunking exists to keep the WORK stack near one volume on
domains where NWORK volumes would not fit; below WORK_FLOOR_BYTES it bought
nothing.
"""
import pytest

from gpuwm.core.morrison_constants import (NWORK, WORK_FLOOR_BYTES,
                                           work_chunk_columns,
                                           work_slot_shape)


def test_the_u2_shape_is_one_call():
    ncol, nz = 156 * 136, 50
    assert work_chunk_columns(ncol, nz) == ncol
    assert work_slot_shape(nz, ncol) == (NWORK * nz * ncol,)


def test_a_big_domain_keeps_its_one_volume_chunks():
    ncol, nz = 1262 * 1224, 49          # the m1200 1 km grid
    assert work_chunk_columns(ncol, nz) == -(-ncol // NWORK)


def test_the_floor_is_bytes_and_never_exceeds_the_domain():
    for nz in (10, 50, 55, 64, 128, 256):
        floor = WORK_FLOOR_BYTES // (NWORK * nz * 4)
        for ncol in (1, 31, floor - 1, floor, floor + 1, 3 * floor, NWORK * floor + 7):
            chunk = work_chunk_columns(ncol, nz)
            assert 1 <= chunk <= ncol
            # One call while the whole domain's stack fits the floor; WRF's
            # one-volume chunks past it (a mid-size domain's scratch stays
            # one volume, not the floor).
            assert chunk == (ncol if ncol <= floor else -(-ncol // NWORK))
            # The WORK stack is the floor or one volume plus NWORK * nz values.
            assert NWORK * nz * chunk * 4 <= max(
                WORK_FLOOR_BYTES, (nz * ncol + NWORK * nz) * 4)


def test_without_nz_the_oracle_chunking_is_unchanged():
    # tests/test_morrison_regime_oracle.py grades ceil(ncol / NWORK) calls.
    assert work_chunk_columns(21216) == -(-21216 // NWORK)
    with pytest.raises(ValueError):
        work_chunk_columns(100, 0)
