"""GPU gate: the Thompson (mp_physics=8) reflectivity route of H_Z(x).

Deliberately its own file, and gpu-marked, structured exactly like the
NSSL gate (``test_da_obsop_nssl_gpu.py``): ``gpuwm.da.obsop``'s CPU test
module imports no CuPy on purpose, and classic Thompson has no float64
column mirror -- the host path names a refusal (tested in
``test_da_obsop.py``), so the ONLY executable H_Z(x) on ``mp_physics=8``
is the CUDA route.  Until this file existed, that route's warrant was
"it is the same ``core.refl`` call the product makes" -- true by
construction, receipted nowhere.  What this file pins:

- the device path genuinely dispatches ``mp_physics=8`` to the product's
  own Thompson ``calc_refl10cm`` port, bit-for-bit against a direct
  ``launch_refl10cm_thompson`` on the same state -- the operator is the
  product authority, not a lookalike;
- the -35 dBZ clear-air floor of the 1/6/8/10 family (NSSL's is 0);
- the operator's hard requirement for the same-call classic
  graupel-number shadow.  That shadow is a REBUILT scratch slot
  (``gpuwm/io/restart.py``: per-call work buffer, never serialized,
  finalized and consumed by REFL_10CM), so a DA caller evaluating H_Z
  between steps cannot read it off a checkpoint -- it must come from the
  microphysics call the analysis time is aligned with.  The refusal is
  the contract the cycling driver has to satisfy; pinning it here keeps
  that wiring gap loud instead of latent.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


pytestmark = pytest.mark.gpu

from gpuwm.core import constants as c        # noqa: E402
from gpuwm.da import obsop                    # noqa: E402


class _DuckState:
    """The minimal state ``simulated_reflectivity`` reads for Thompson.

    ``scratch`` mimics ``DomainState.scratch``: a persistent, reused slot
    keyed by name, which is what lets the operator return the state's own
    ``refl_10cm`` buffer.
    """

    def __init__(self):
        self._slots = {}

    def scratch(self, shape, slot, dtype=None):
        import cupy as cp

        dtype = cp.float32 if dtype is None else dtype
        buf = self._slots.get(slot)
        if buf is None or buf.shape != tuple(shape) or buf.dtype != dtype:
            buf = cp.zeros(shape, dtype=dtype)
            self._slots[slot] = buf
        return buf


def _thompson_state(shape=(4, 4, 4), *, precipitating=True):
    """A moist state carrying Thompson's field set plus Morrison's moments.

    The Morrison extras (``ns``/``ng``) exist so the not-the-Morrison-route
    negative control can run the SAME state through ``mp_physics=10``; the
    Thompson kernel reads none of them.
    """
    import cupy as cp

    nz, ny, nx = shape

    def dev(value):
        return cp.asarray(np.full(shape, value, dtype=np.float32))

    state = _DuckState()
    state.p = dev(8.0e4)
    state.thp = dev(0.0)
    state.thb = cp.asarray(
        np.full((nz,), 290.0 * (c.P0 / 8.0e4) ** c.RCP, dtype=np.float32))
    state.qv = dev(1.0e-2)
    mass = 1.0e-3 if precipitating else 0.0
    for name in ("qr", "qs", "qg"):
        setattr(state, name, dev(mass))
    number = 1.0e4 if precipitating else 0.0
    for name in ("nr", "ns", "ng"):
        setattr(state, name, dev(number))
    shadow = dev(1.0e3 if precipitating else 0.0)
    return state, shadow


def test_mp8_dispatches_to_the_product_calc_refl10cm_bit_for_bit():
    """The DA route is the SAME diagnostic the microphysics adapter runs.

    Bit-for-bit against a direct ``launch_refl10cm_thompson`` on the same
    state, with the standalone-path temperature diagnosis
    (``(thb + thp) * (p/P0)**RCP`` in float32) that
    ``compute_refl_10cm`` performs when the microphysics-time t1d/p1d
    pair is omitted -- the right choice for a DA operator evaluated
    between steps, and the exact arithmetic is part of the receipt.
    """
    import cupy as cp

    from gpuwm.core.refl import launch_refl10cm_thompson
    from gpuwm.core.state import DTYPE

    state, shadow = _thompson_state()
    cfg = SimpleNamespace(mp_physics=8)

    got = obsop.simulated_reflectivity(
        state, cfg, thompson_graupel_number=shadow)
    cp.cuda.Stream.null.synchronize()
    assert got.dtype == cp.float32
    assert got.shape == (4, 4, 4)
    got = got.copy()  # the state owns the refl_10cm slot; keep our frame

    thb = state.thb[:, None, None]
    temperature = (thb + state.thp) * cp.power(
        state.p / DTYPE(c.P0), DTYPE(c.RCP))
    reference = cp.zeros(got.shape, dtype=cp.float32)
    launch_refl10cm_thompson(
        state.qv, state.qr, state.nr, state.qs, state.qg,
        shadow, temperature, state.p, reference)
    cp.cuda.Stream.null.synchronize()
    cp.testing.assert_array_equal(got, reference)
    # An all-floor field would pass assert_array_equal vacuously; prove
    # the precipitating state actually exercised the Rayleigh sums.
    assert float(got.max()) > -35.0


def test_mp8_floors_at_minus_thirty_five_dbz_in_clear_air():
    """The 1/6/8/10 family floor, pinned on the Thompson route.

    NSSL floors at 0 dBZ; Thompson at -35.  An innovation computed
    against obs carrying the wrong clear-air floor manufactures a 35 dB
    difference out of two agreeing clear skies, so the floor is part of
    the operator's contract, not a rendering detail.
    """
    import cupy as cp

    clear, shadow = _thompson_state(precipitating=False)
    got = obsop.simulated_reflectivity(
        clear, SimpleNamespace(mp_physics=8), thompson_graupel_number=shadow)
    cp.cuda.Stream.null.synchronize()
    assert float(got.min()) == -35.0
    assert float(got.max()) == -35.0


def test_mp8_requires_the_same_call_graupel_number_shadow():
    """No shadow, no reflectivity -- a refusal, never a silent guess.

    WRF classic Thompson's ng1d is private to the scheme: a per-call
    REBUILT scratch, absent from checkpoints by design.  A DA caller
    that cannot produce it must hear that loudly, because the tempting
    fallbacks (zeros, or Morrison's prognostic ng) are both a different
    formulation wearing the right name.
    """
    state, _ = _thompson_state()
    with pytest.raises(ValueError, match="graupel number shadow"):
        obsop.simulated_reflectivity(state, SimpleNamespace(mp_physics=8))


def test_mp8_is_not_the_morrison_route():
    """Same state, mp_physics flipped: the answers must differ.

    Bit-for-bit against the Thompson kernel proves the dispatch landed
    somewhere exact; this proves it landed somewhere SPECIFIC -- the same
    fields through Morrison's ``refl10cm_hm`` (two-moment snow/graupel,
    its own PSD intercepts) give a different field, so a dispatch bug
    that mapped mp=8 onto mp=10 could not pass both tests.
    """
    import cupy as cp

    state, shadow = _thompson_state()
    z8 = obsop.simulated_reflectivity(
        state, SimpleNamespace(mp_physics=8),
        thompson_graupel_number=shadow).copy()
    z10 = obsop.simulated_reflectivity(
        state, SimpleNamespace(mp_physics=10, morr_rimed_ice=1)).copy()
    cp.cuda.Stream.null.synchronize()
    assert bool((z8 != z10).any())
