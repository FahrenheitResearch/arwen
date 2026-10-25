"""Morrison's WORK stack is sized for one column call, not the domain.

The WRF v4.6.1 transcription (kernels/morrison.cu) keeps WRF's structure:
the process stage stores NWORK per-cell planes (tendencies, temperature,
CPM, LAMI, ACN, fallout tendencies) for sedimentation and the final loop.
Carried at full size that stack added 19 volumes of VRAM to the
door-default microphysics: on the d01 itemization fixture the scheme's
scratch rose from 596,974,580 to 783,374,580 bytes and domains the card
had held stopped fitting.  A forecast step therefore runs the scheme in
``work_chunk_columns(ncol)``-column calls (WRF's columns are independent;
tests/test_morrison_regime_oracle.py grades the chunked calls against WRF
word for word), and the WORK slot holds one call's columns: one volume
plus at most NWORK * nz values -- or, on a domain small enough that NWORK
volumes are under WORK_FLOOR_BYTES (128 MiB), the whole domain in one call:
the chunks bought no memory there and cost the u2 user shape 21 launches of
three kernels a step (WOOF 1.1 final canary, 1.24x its 2.8.7 twin).  CPU-only.
"""

from __future__ import annotations

from gpuwm.config import RunConfig
from gpuwm.core.morrison_constants import (NWORK, SCRATCH_VOLUME_SLOTS,
                                           SCRATCH_WORK_SLOT, WORK_FLOOR_BYTES,
                                           work_chunk_columns,
                                           work_slot_shape)

FLOOR_VALUES = WORK_FLOOR_BYTES // 4


def test_forecast_chunking_keeps_work_at_one_volume():
    for nz, ncol in ((50, 106), (50, 1), (80, 1799 * 1059), (64, 21),
                     (64, 22), (32, 41 * 40)):
        chunk = work_chunk_columns(ncol, nz)
        assert 1 <= chunk <= ncol
        assert -(-ncol // chunk) <= NWORK          # at most NWORK calls
        (size,) = work_slot_shape(nz, ncol)
        assert size == NWORK * nz * chunk
        # one volume plus slack, or the 128 MiB floor on a small domain
        assert size < max(nz * ncol + NWORK * nz, FLOOR_VALUES + NWORK * nz)
    # A domain past the floor keeps WRF-sized one-volume chunks.
    assert work_chunk_columns(1799 * 1059, 80) == -(-(1799 * 1059) // NWORK)


def test_preflight_prices_the_chunked_work_slot():
    from gpuwm.core.preflight import scratch_slot_registry

    cfg = RunConfig(nx=41, ny=32, nz=40, dx=3000.0, dy=3000.0, ztop=20000.0,
                    dt=18.0, run_seconds=180.0, mp_physics=10)
    slots = scratch_slot_registry(cfg)
    ny, nx = 32, 41
    assert slots[SCRATCH_WORK_SLOT] == work_slot_shape(cfg.nz, ny * nx)
    volume = cfg.nz * ny * nx
    # The four preparation fields, WORK and refl_t: no more than the six
    # volumes the replaced kernel drew (theta, rho, pii, dz, ice_to_snow,
    # refl_t), plus WORK's rounding slack.
    morrison = [slots[name] for name in SCRATCH_VOLUME_SLOTS] + [
        slots[SCRATCH_WORK_SLOT], slots["refl_t"]]
    values = sum(_count(shape) for shape in morrison)
    # (A small domain's WORK may take the whole domain, up to the floor.)
    assert values <= 6 * volume + NWORK * cfg.nz + FLOOR_VALUES


def _count(shape):
    out = 1
    for extent in shape:
        out *= int(extent)
    return out
