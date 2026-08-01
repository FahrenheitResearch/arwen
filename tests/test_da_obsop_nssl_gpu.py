"""GPU gate: the NSSL (mp_physics=18) reflectivity route of H_Z(x).

Deliberately its own file, and gpu-marked: ``gpuwm.da.obsop``'s CPU test
module imports no CuPy on purpose, but the NSSL diagnostic is CUDA only --
five ice categories with their own moments make it a scheme port rather
than an adapter over one of the mirrored three, so there is no float64
column mirror and the host path names a refusal instead (that half is
tested in ``test_da_obsop.py``).  What this file pins is that the device
path genuinely dispatches ``mp_physics=18`` to the product's own
``radardd02``, not to Morrison or Thompson.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


pytestmark = pytest.mark.gpu

from gpuwm.core import constants as c        # noqa: E402
from gpuwm.da import obsop                    # noqa: E402


class _DuckState:
    """The minimal state ``simulated_reflectivity`` reads for NSSL.

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


def _nssl_state(shape=(4, 4, 4), *, precipitating=True):
    import cupy as cp

    nz, ny, nx = shape

    def dev(value):
        return cp.asarray(np.full(shape, value, dtype=np.float32))

    state = _DuckState()
    state.p = dev(8.0e4)
    state.alt = dev(1.0)                       # rho = 1/alt = 1 kg/m3
    state.thp = dev(0.0)
    state.thb = cp.asarray(
        np.full((nz,), 290.0 * (c.P0 / 8.0e4) ** c.RCP, dtype=np.float32))
    state.qv = dev(1.0e-2)
    mass = 1.0e-3 if precipitating else 0.0
    for name in ("qr", "qi", "qs", "qg", "qh"):
        setattr(state, name, dev(mass))
    number = 1.0e4 if precipitating else 0.0
    for name in ("qnr", "qni", "qns", "qng", "qnh"):
        setattr(state, name, dev(number))
    volume = 2.0e-6 if precipitating else 0.0
    for name in ("qvolg", "qvolh"):
        setattr(state, name, dev(volume))
    return state


def test_mp18_dispatches_to_the_product_radardd02_bit_for_bit():
    """The DA route is the SAME diagnostic the production coordinator runs.

    Bit-for-bit against a direct ``launch_radardd02`` on the same state,
    with the per-mass moment convention and the ``rho = 1/alt`` density the
    NSSL runtime uses -- so the operator is proven to be the product
    authority, not a lookalike.
    """
    import cupy as cp

    from gpuwm.core.nssl2_diagnostics import launch_radardd02

    state = _nssl_state()
    cfg = SimpleNamespace(mp_physics=18)

    got = obsop.simulated_reflectivity(state, cfg)
    cp.cuda.Stream.null.synchronize()
    assert got.dtype == cp.float32
    assert got.shape == (4, 4, 4)

    rho = cp.float32(1.0) / state.alt
    temperature = (state.thb[:, None, None] + state.thp) * cp.power(
        state.p / np.float32(c.P0), np.float32(c.RCP))
    reference = cp.empty(got.shape, dtype=cp.float32)
    launch_radardd02(
        rho, temperature,
        state.qr, state.qi, state.qs, state.qg, state.qh,
        state.qnr, state.qni, state.qns, state.qng, state.qnh,
        state.qvolg, state.qvolh, reference,
        output_due=True, concentration_space=False)
    cp.cuda.Stream.null.synchronize()
    cp.testing.assert_array_equal(got, reference)


def test_mp18_floors_at_zero_dbz_not_minus_thirty_five():
    """radardd02's native floor is 0 dBZ, and the caller has to know it.

    The 1/6/8/10 routes floor at -35; NSSL does not.  Clear air in an NSSL
    H(x) column reads 0 dBZ, so differencing it against a -35 dBZ
    observation floor would manufacture a 35 dB innovation out of two
    agreeing clear skies.  Pinning the floor here is what makes that
    documented rather than latent.
    """
    import cupy as cp

    clear = _nssl_state(precipitating=False)
    got = obsop.simulated_reflectivity(clear, SimpleNamespace(mp_physics=18))
    cp.cuda.Stream.null.synchronize()
    assert float(got.min()) == 0.0
    assert float(got.max()) == 0.0


def test_mp18_refuses_the_t1d_p1d_pair_it_does_not_use():
    """radardd02 diagnoses its own density; it is not the t1d/p1d diagnostic.

    Passing the microphysics-time pair the 1/6/8/10 routes accept is a
    caller error worth naming, not a silent no-op that looks like it was
    honoured.
    """
    import cupy as cp

    state = _nssl_state()
    with pytest.raises(ValueError, match="radardd02"):
        obsop.simulated_reflectivity(
            state, SimpleNamespace(mp_physics=18),
            temperature=state.p, pressure=state.p)
    cp.cuda.Stream.null.synchronize()
