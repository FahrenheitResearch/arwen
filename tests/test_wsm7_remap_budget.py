"""WSM7's PLM sedimentation remap conserves mass (declared divergence D1).

THE BREAKAGE THIS PREVENTS
--------------------------
WRF v4.6.1's nislfv_rain_plm/plm6 search for the arrival interface above
each target layer (find_kt) stops one interface short.  A layer whose top
lies above every interface it tests falls through to kt = kb - 1, neither
remap branch runs, and the layer's arriving mass is deleted -- in a
transport-only column (0.001 kg m-3, 100 m layers, 1 m/s) the top layer's
share vanishes on every call while the surface sees none of it (queue 073;
WSM6 and WDM6 carry the same search).  WOOF's kernel searches every arrival
interface and the column top and never leaves kt below kb, so the remap
conserves: final column plus fallout equals the initial inventory.

The check replays that control through the kernel's own remap
(``wsm7_remap_probe`` in kernels/wsm7.cu, the very device function the
scheme sediments with) at 2, 50 and 80 levels and dt 0.001, 1 and 20 s, and
a planted reversion of the search -- WRF's loop substituted back into the
same source between the REMAP-SEARCH markers -- must fail it.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu

LEVELS = (2, 50, 80)
STEPS = (0.001, 1.0, 20.0)
#: WRF's stock search, as substituted by the planted reversion.
STOCK_SEARCH = """\
            {
                for (int kk = kt; kk < km; ++kk)
                    if (zi[k + 1] <= za[kk]) { kt = kk; break; }
                kt = kt - 1;
            }
"""


def _source(planted: bool) -> str:
    from gpuwm.core.kernels import module_source_int_defines

    src = module_source_int_defines("wsm7", (("WSM7_KMAX", 80),))
    if planted:
        head, rest = src.split("// REMAP-SEARCH BEGIN", 1)
        _, tail = rest.split("// REMAP-SEARCH END", 1)
        src = head + "// REMAP-SEARCH BEGIN (planted: WRF's search)\n" \
            + STOCK_SEARCH + "            // REMAP-SEARCH END" + tail
    return src


def _probe(src: str, km: int, dt: float):
    import cupy as cp
    from cupy.cuda import compiler
    from gpuwm.core.wsm_family import COMPILE_OPTIONS

    ptx, _ = compiler.compile_using_nvrtc(src, COMPILE_OPTIONS, None, "wsm7.cu")
    module = cp.cuda.function.Module()
    module.load(ptx.encode() if isinstance(ptx, str) else ptx)
    fn = module.get_function("wsm7_remap_probe")
    dz = cp.full(km, 100.0, cp.float32)
    den = cp.ones(km, cp.float32)
    ww = cp.ones(km, cp.float32)
    rq = cp.full(km, 1.0e-3, cp.float32)
    precip = cp.zeros(1, cp.float32)
    fn((1,), (1,), (dz, den, ww, rq, precip, np.float32(dt), np.int32(km)))
    before = 1.0e-3 * 100.0 * km
    after = float(cp.sum(rq.astype(cp.float64) * 100.0)) + float(precip[0])
    return before, after


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("km", LEVELS)
@pytest.mark.parametrize("dt", STEPS)
def test_the_remap_conserves_column_plus_fallout(km, dt):
    before, after = _probe(_source(planted=False), km, dt)
    assert abs(after - before) / before < 1.0e-6, (km, dt, before, after)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("km", LEVELS)
def test_a_planted_reversion_of_the_search_fails_the_budget(km):
    before, after = _probe(_source(planted=True), km, 1.0)
    assert abs(after - before) / before > 1.0e-3, (km, before, after)


def test_the_planted_substitution_reaches_the_search():
    """CPU guard: the markers exist once and the planted source differs, so
    the GPU reversion test cannot pass vacuously."""
    src = _source(planted=False)
    assert src.count("// REMAP-SEARCH BEGIN") == 1
    assert src.count("// REMAP-SEARCH END") == 1
    planted = _source(planted=True)
    assert "kk <= km" in src and "kk <= km" not in planted
    assert STOCK_SEARCH in planted
