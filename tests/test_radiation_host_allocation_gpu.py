"""A legacy-RRTMG radiation call allocates no per-column host memory.

Named breakage (1 km HRRR-physics run, 2026-10-05, 3x RTX PRO 6000 on a
shared host, engine a80bc6b21): five radiation steps took 29, 205, 268, 402
and 48 s while the cards sat idle and ordinary steps were untouched.  A
radiation step was the only step that allocated large fresh host arrays:
each rank interpolated the ozone climatology to every column on the host
(ncol x 59 float32) and copied the 3-D ozone back to the host (nz x ncol
float32), about 370 MB per rank per call at 845k columns.  On box G the same
run showed 9x the page-fault rate, 40x the transparent-huge-page faults and
14x the direct-compaction stalls inside radiation steps; on the loaded host
that page-fault path stalled for minutes.

The contract this pins: the host memory one radiation call allocates does
not grow with the column count beyond a few bytes per column (2-D carriers
such as a host coszen copy are tolerated; any per-column profile is not).
Measured with tracemalloc, which NumPy reports its data buffers to, as the
slope of the call's peak host allocation between two grid sizes.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from conftest import HAS_GPU

import test_rrtmg_legacy_wiring as wiring

#: Per-column host growth a call may show, in bytes.  Eight float32 per
#: column covers every 2-D host carrier the driver may legitimately touch;
#: one 50-level profile is 200 bytes, the ozone pair it replaces ~440.
MAX_BYTES_PER_COLUMN = 32

GRIDS = ((60, 80), (120, 160))


def _call_peak(profile, ny, nx):
    from gpuwm.core.rrtmg_legacy import RRTMGLegacyRadiation

    ncol = ny * nx
    bundle = wiring._bundle(profile, ncol, seed=20261005)
    lat = np.linspace(25.0, 50.0, ncol).astype(np.float32)
    lon = np.linspace(-120.0, -75.0, ncol).astype(np.float32)
    env = wiring._env_from(bundle, np.arange(ncol), ny, nx, lat, lon)
    adapter = RRTMGLegacyRadiation(
        wiring.START, env.lat.reshape(ny, nx), env.lon.reshape(ny, nx),
        p_top=env.p_top)
    # Two warm calls: kernel loads, table uploads and any one-time host
    # staging belong to construction or the first call, not to the per-call
    # cost this contract prices.
    for _ in range(2):
        wiring._call(adapter, env)
    wiring.cp.cuda.runtime.deviceSynchronize()
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        base, _ = tracemalloc.get_traced_memory()
        result = wiring._call(adapter, env)
        wiring.cp.cuda.runtime.deviceSynchronize()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    del result
    return ncol, env.nz, peak - base


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU / cupy")
def test_radiation_call_host_allocation_does_not_scale_with_columns(profile):
    (n0, nz, p0), (n1, _, p1) = (_call_peak(profile, ny, nx)
                                 for ny, nx in GRIDS)
    slope = (p1 - p0) / float(n1 - n0)
    assert slope <= MAX_BYTES_PER_COLUMN, (
        f"one legacy-RRTMG call allocates {slope:.0f} host bytes per column "
        f"(peak {p0 / 2**20:.1f} MiB at {n0} columns, {p1 / 2**20:.1f} MiB at "
        f"{n1}; nz={nz}); a per-column host profile is back on the radiation "
        f"path, the allocation that stalled 1 km radiation steps for minutes "
        f"on a loaded host")


profile = wiring.profile
