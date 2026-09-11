"""GPU gate: the Milbrandt-Yau (mp_physics=9) reflectivity route of H_Z(x).

Its own file, and gpu-marked, for the reason
``test_da_obsop_nssl_gpu.py`` is: the scheme's Z is CUDA only, there is no
float64 column mirror in ``gpuwm.verify.npref``, and the host path names a
refusal instead.

What this file proves is the claim that made mp=9 an observation operator
at all (audit R-015 / R-050): the Z block of
``mp_milbrandt2mom_main``'s final diagnostics is SEPARABLE from the state
update it sits beside.  ``gpuwm/core/kernels/milbrandt2_zet.cu`` is a
transcription of that block, and the first test drives BOTH kernels over
the same random state and compares the two dBZ fields bitwise -- so the
duplication the loader forces (``cupy.RawModule`` has no ``#include``) is
held to the original by measurement rather than by review, and the
operator cannot quietly become a fourth Z formula.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


pytestmark = pytest.mark.gpu

from gpuwm.core import constants as c        # noqa: E402
from gpuwm.da import obsop                   # noqa: E402


class _DuckState:
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


def _my2_state(shape=(6, 5, 4), *, precipitating=True, seed=20260910):
    """A Milbrandt-Yau state with per-MASS number moments, as a run carries."""
    import cupy as cp

    nz, ny, nx = shape
    rng = np.random.default_rng(seed)

    def dev(array):
        return cp.asarray(np.ascontiguousarray(array, dtype=np.float32))

    state = _DuckState()
    pressure = np.linspace(9.5e4, 2.0e4, nz)[:, None, None] * np.ones(shape)
    state.p = dev(pressure)
    state.alt = dev(np.full(shape, 1.0))
    state.thp = dev(rng.uniform(-2.0, 2.0, shape))
    state.thb = cp.asarray(
        np.linspace(290.0, 340.0, nz).astype(np.float32))
    state.qv = dev(np.full(shape, 5.0e-3))
    for name in ("qc", "qr", "qi", "qs", "qg", "qh"):
        values = (rng.uniform(0.0, 3.0e-3, shape) if precipitating
                  else np.zeros(shape))
        setattr(state, name, dev(values))
    for name in ("nc", "nr", "ni", "ns", "ng", "nh"):
        values = (rng.uniform(1.0e2, 1.0e6, shape) if precipitating
                  else np.zeros(shape))
        setattr(state, name, dev(values))
    return state


def _reference_zet(state):
    """The scheme's OWN Z block, launched through milbrandt2_diagnostics.

    The diagnostics kernel takes numbers per unit VOLUME (the convention
    that holds inside a scheme call), so the state's per-mass moments are
    converted with the same ``de`` the operator forms.  Nothing else about
    the call differs, which is what makes a bitwise comparison meaningful.
    """
    import cupy as cp

    from gpuwm.core.kernels import get_kernel
    from gpuwm.core.milbrandt2 import _constants_device

    shape = tuple(state.p.shape)
    nz, ny, nx = shape
    temperature = (state.thb[:, None, None] + state.thp) * cp.power(
        state.p / np.float32(c.P0), np.float32(c.RCP))
    de = state.p / (np.float32(287.05) * temperature)
    numbers = [cp.ascontiguousarray(getattr(state, name) * de)
               for name in ("nc", "nr", "ni", "ns", "ng", "nh")]
    qv = cp.ascontiguousarray(state.qv.copy())
    out = cp.zeros(shape, dtype=cp.float32)
    ncell = nz * ny * nx
    threads = 64
    get_kernel("milbrandt2", "milbrandt2_diagnostics")(
        (((ncell + threads - 1) // threads),), (threads,), (
            cp.ascontiguousarray(temperature), qv,
            state.qc, state.qr, state.qi, state.qs, state.qg, state.qh,
            *numbers,
            cp.ascontiguousarray(state.p), out, _constants_device(),
            np.int32(nz), np.int32(ny), np.int32(nx)))
    cp.cuda.Stream.null.synchronize()
    return out


def test_the_operator_is_the_schemes_own_z_block_bitwise():
    import cupy as cp

    state = _my2_state()
    cfg = SimpleNamespace(mp_physics=9)
    got = obsop.simulated_reflectivity(state, cfg)
    cp.cuda.Stream.null.synchronize()
    assert got.dtype == cp.float32 and got.shape == (6, 5, 4)
    cp.testing.assert_array_equal(got, _reference_zet(state))


def test_the_operator_updates_nothing_it_reads():
    """H(x) means H(x): the background must come back unchanged.

    This is the criterion that separates mp=9 from mp=50 in
    ``NATIVE_Z_NOT_SEPARABLE_FROM_THE_STEP``.  If the lifted block ever
    grew the diagnostics kernel's ``Q`` clamp or its ``#/m3 -> #/kg``
    write-back, the operator would move the very state it observes and
    this test is what says so.
    """
    import cupy as cp

    state = _my2_state()
    before = {name: getattr(state, name).copy()
              for name in ("qv", "qc", "qr", "qi", "qs", "qg", "qh",
                           "nc", "nr", "ni", "ns", "ng", "nh", "p", "thp")}
    obsop.simulated_reflectivity(state, SimpleNamespace(mp_physics=9))
    cp.cuda.Stream.null.synchronize()
    for name, original in before.items():
        cp.testing.assert_array_equal(getattr(state, name), original)


def test_clear_air_reads_the_tables_minus_ninety_nine():
    """A hydrometeor-free column reads exactly the table's floor.

    ``CLEAR_AIR_FLOOR_DBZ[9]`` is -99.0 and the whole point of the row is
    that it is NOT the -35 the refl10cm family floors at, so the number is
    read off the operator rather than trusted.
    """
    import cupy as cp

    state = _my2_state(precipitating=False)
    got = obsop.simulated_reflectivity(state, SimpleNamespace(mp_physics=9))
    cp.cuda.Stream.null.synchronize()
    assert float(cp.asnumpy(got).min()) == -99.0
    assert float(cp.asnumpy(got).max()) == -99.0
    assert obsop.clear_air_floor_dbz(9) == -99.0
