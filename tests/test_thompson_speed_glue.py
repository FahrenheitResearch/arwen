"""The fused classic adapter launches against the sequence they replaced.

``launch_adapter_prepare``, ``_entry``, ``_masks`` and ``_finish`` (the
``thompson_adapter_*`` kernels) must write exactly what the former CuPy
operations, ``save_pre_mp_theta``, ``moist_physics_finish`` and the four small
kernels wrote.  Both run on the same seeded state, with signed zeros, values
at R1, NaN and infinities mixed in, and every output is compared as raw words.
"""
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

SHAPE = (9, 7, 11)


def _cupy():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("no CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("no CUDA device")
    return cp


def _words(cp, a, b, name):
    np.testing.assert_array_equal(cp.asnumpy(a).view(np.uint32),
                                  cp.asnumpy(b).view(np.uint32),
                                  err_msg=name)


def _state(cp, full_base: bool):
    rng = np.random.default_rng(19)
    nz, ny, nx = SHAPE
    s = SimpleNamespace()
    for name in ("qc", "qi", "qr", "qs", "qg"):
        a = rng.uniform(0.0, 2.0e-4, SHAPE).astype(np.float32)
        a.ravel()[::5] = np.float32(1.0e-12)
        a.ravel()[1::7] = np.float32(-0.0)
        a.ravel()[2::13] = np.float32(np.nan)
        setattr(s, name, cp.asarray(a))
    for name in ("ni", "nr"):
        a = rng.uniform(1.0, 1.0e5, SHAPE).astype(np.float32)
        a.ravel()[3::11] = np.float32(-0.0)
        setattr(s, name, cp.asarray(a))
    s.p = cp.asarray(rng.uniform(2.0e4, 1.0e5, SHAPE).astype(np.float32))
    # Dry cells (no condensate at all) exercise the supersaturation branch
    # of the column flag on both sides of 0 C.
    s.qv = cp.asarray(rng.uniform(1.0e-5, 0.02, SHAPE).astype(np.float32))
    for name in ("qc", "qi", "qr", "qs", "qg"):
        getattr(s, name)[:, 0, :] = 0.0
    s.qv[:, 0, :] = 1.0e-7
    s.qv[3, 0, 5] = 0.02
    thb = np.full(SHAPE if full_base else (nz,), 300.0, np.float32)
    s.thb = cp.asarray(thb)
    s.thp = cp.asarray(rng.uniform(-40.0, 20.0, SHAPE).astype(np.float32))
    phb = np.arange(nz + 1, dtype=np.float32) * 2000.0
    s.phb = cp.asarray(np.broadcast_to(phb[:, None, None], (nz + 1, ny, nx)).copy()
                       if full_base else phb)
    s.php = cp.asarray(rng.uniform(-10.0, 10.0, (nz + 1, ny, nx)).astype(np.float32))
    s.h_diabatic = cp.zeros(SHAPE, cp.float32)
    return s


def _scratch(cp):
    nz, ny, nx = SHAPE
    volume = {name: cp.full(SHAPE, np.float32(7.0))
              for name in ("th", "pii", "t", "dz", "entry_g", "warm", "ng")}
    surface = {name: cp.full((ny, nx), np.float32(7.0))
               for name in ("gncv", "micro", "rain", "graupel")}
    return volume, surface


@pytest.mark.parametrize("full_base", [False, True])
def test_prepare_entry_and_masks_match_the_former_sequence(full_base):
    cp = _cupy()
    from gpuwm.core import constants as c
    from gpuwm.core import thompson
    from gpuwm.core.microphysics import save_pre_mp_theta

    a, b = _state(cp, full_base), _state(cp, full_base)
    va, sa = _scratch(cp)
    vb, sb = _scratch(cp)

    # The adapter keeps CuPy's power for the Exner function.
    va["pii"][...] = cp.power(a.p / np.float32(c.P0), np.float32(c.RCP))
    thompson.launch_adapter_prepare(
        a.thb, a.thp, a.phb, a.php, va["th"], va["pii"], va["t"],
        va["dz"], a.h_diabatic, a.qc, a.qi, a.ni, a.qr, a.nr, a.qs, a.qg,
        va["entry_g"], va["warm"], sa["gncv"], sa["micro"])
    thompson.launch_adapter_entry(
        a.qc, a.qi, a.qr, a.qs, a.qg, va["t"], a.p, a.qv, va["ng"],
        sa["micro"])

    # The former sequence (gpuwm/core/microphysics.py:_apply_thompson at
    # d6929cb8d).
    thb = b.thb if b.thb.ndim == 3 else b.thb[:, None, None]
    phb = b.phb if b.phb.ndim == 3 else b.phb[:, None, None]
    vb["th"][...] = thb + b.thp
    vb["pii"][...] = cp.power(b.p / np.float32(c.P0), np.float32(c.RCP))
    vb["t"][...] = vb["th"] * vb["pii"]
    z8w = (phb + b.php) / np.float32(c.G)
    vb["dz"][...] = z8w[1:] - z8w[:-1]
    cp.greater(b.qg, np.float32(1.0e-12), out=vb["entry_g"])
    cp.greater_equal(vb["t"], np.float32(273.15), out=vb["warm"])
    sb["gncv"].fill(np.float32(0.0))
    save_pre_mp_theta(b)
    for mass, number in ((b.qc, None), (b.qi, b.ni), (b.qr, b.nr),
                         (b.qs, None), (b.qg, None)):
        present = mass > np.float32(1.0e-12)
        if number is not None:
            number[...] = cp.where(present, number, np.float32(0.0))
        mass[...] = cp.where(present, mass, np.float32(0.0))
    thompson.launch_microphysics_columns(
        b.qc, b.qi, b.qr, b.qs, b.qg, vb["t"], b.p, b.qv, sb["micro"])
    thompson.launch_classic_graupel_number_init(
        b.qg, vb["t"], b.p, b.qv, vb["ng"])

    for name in va:
        _words(cp, va[name], vb[name], name)
    for name in ("gncv", "micro"):
        _words(cp, sa[name], sb[name], name)
    for name in ("qc", "qi", "ni", "qr", "nr", "qs", "qg", "h_diabatic"):
        _words(cp, getattr(a, name), getattr(b, name), name)
    assert 0.0 < float(sb["micro"].mean()) < 1.0

    # Post-source masks on the rewritten state.
    thompson.launch_adapter_masks(a.qr, a.qg, va["entry_g"], sa["rain"],
                                  sa["graupel"])
    thompson.launch_hydrometeor_column_mask(b.qr, sb["rain"])
    thompson.launch_graupel_fallout_column_mask(vb["entry_g"], b.qg,
                                                sb["graupel"])
    for name in ("rain", "graupel"):
        _words(cp, sa[name], sb[name], name)


@pytest.mark.parametrize("no_mp_heating", [0, 1])
def test_finish_matches_the_former_sequence(no_mp_heating):
    cp = _cupy()
    from gpuwm.core import thompson
    from gpuwm.core.microphysics import moist_physics_finish, save_pre_mp_theta

    nz, ny, nx = SHAPE
    cfg = SimpleNamespace(mp_tend_lim=0.1, no_mp_heating=no_mp_heating)
    dt = 6.0

    def make():
        s = _state(cp, False)
        save_pre_mp_theta(s)
        pii = cp.asarray(rng.uniform(0.6, 1.0, SHAPE).astype(np.float32))
        t = (s.thb[:, None, None] + s.thp) * pii
        # Increments inside and beyond the clamp, and special values.
        t += cp.asarray(rng.uniform(-1.0, 1.0, SHAPE).astype(np.float32))
        t.ravel()[:5] = cp.asarray(
            [np.nan, np.inf, -np.inf, 0.0, -0.0], dtype=cp.float32)
        rain = np.resize(np.array([0.0, -0.0, np.nan, 1.0e-12, 2.0e-12, 1.0,
                                   np.inf, 3.0e-5], np.float32), (ny, nx))
        surface = [cp.asarray(rain)] + [
            cp.asarray(rng.uniform(0.0, 2.0e-5, (ny, nx)).astype(np.float32))
            for _ in range(2)]
        surface[1].ravel()[:3] = cp.asarray([np.nan, -0.0, np.inf],
                                            dtype=cp.float32)
        return s, pii, t, surface

    rng = np.random.default_rng(23)
    a, pii_a, t_a, (rain_a, snow_a, graupel_a) = make()
    rng = np.random.default_rng(23)
    b, pii_b, t_b, (rain_b, snow_b, graupel_b) = make()
    th_a, th_b = cp.empty_like(t_a), cp.empty_like(t_b)
    sr_a, sr_b = cp.empty_like(rain_a), cp.empty_like(rain_b)

    thompson.launch_adapter_finish(t_a, pii_a, th_a, a.thp, a.h_diabatic,
                                   rain_a, snow_a, graupel_a, sr_a, cfg, dt)

    th_b[...] = t_b / pii_b
    moist_physics_finish(b, cfg, th_b, dt)
    frozen = snow_b + graupel_b
    sr_b[...] = cp.where(rain_b > np.float32(1.0e-12),
                         cp.minimum(np.float32(1.0), frozen / rain_b),
                         np.float32(0.0))

    _words(cp, th_a, th_b, "th")
    _words(cp, a.thp, b.thp, "thp")
    _words(cp, a.h_diabatic, b.h_diabatic, "h_diabatic")
    _words(cp, sr_a, sr_b, "sr")
